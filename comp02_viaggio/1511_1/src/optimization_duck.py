import optuna
import gc
import lightgbm as lgb
import duckdb
import numpy as np
import logging
import json
import os
from sklearn.model_selection import KFold
from datetime import datetime
from .config import *
from .gain_function import *

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, conn, tabla: str) -> float:
    """
    Función objetivo con 5-fold CV estándar.
    Usa períodos diferentes para clase 0 y clase 1.
    Entrena con target_binario, evalúa con target_ternario.
    """
    # Hiperparámetros a optimizar
    num_leaves = trial.suggest_int('num_leaves', 8, 50) 
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 50, 500) 
    feature_fraction = trial.suggest_float('feature_fraction', 0.3, 0.8) 
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.6, 1.0) 
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    reg_alpha = trial.suggest_float('reg_alpha', 0.1, 10.0, log=True) 
    reg_lambda = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True) 
    max_depth = trial.suggest_int('max_depth', 3, 20) 

    params = {
        'objective': 'binary',
        'metric': 'custom',
        'boosting_type': 'gbdt',
        'max_bin': 31,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'learning_rate': learning_rate,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'max_depth': max_depth,
        'is_unbalance': True,
        'boost_from_average': True,
        'feature_pre_filter': True,
        'bagging_freq': 1,
        'n_jobs': -1,
        'seed': SEMILLAS[0],
        'verbose': -1
    }
    
    # Obtener dataset completo UNA SOLA VEZ
    periodos_clase1_str = ','.join(map(str, PERIODOS_CLASE_1))
    periodos_clase0_str = ','.join(map(str, PERIODOS_CLASE_0))
    
    query_completo = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_clase0_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {SEMILLAS[0]})
        ),
        clase_1_completa AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_clase1_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    data = conn.execute(query_completo).fetchnumpy()
    
    # Estadísticas del dataset completo
    n_total = len(data['target_binario'])
    n_clase_0 = (data['target_binario'] == 0).sum()
    n_clase_1 = (data['target_binario'] == 1).sum()
    
    logger.info(f"Trial {trial.number} - Dataset completo:")
    logger.info(f"  Total: {n_total:,} registros")
    logger.info(f"  Clase 0: {n_clase_0:,} ({n_clase_0/n_total*100:.1f}%)")
    logger.info(f"  Clase 1: {n_clase_1:,} ({n_clase_1/n_total*100:.1f}%)")
    logger.info(f"  Ratio: {n_clase_1/n_clase_0:.3f}:1")
    
    # Preparar features
    feature_cols = [col for col in data.keys() 
                   if col not in ['target_binario', 'target_ternario', 'foto_mes']]
    
    X = np.column_stack([data[col] for col in feature_cols])
    y_binario = data['target_binario']
    y_ternario = data['target_ternario']
    
    # 5-Fold CV
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEMILLAS[0])
    
    ganancias_folds = []
    best_iterations = []
    stats_folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        logger.info(f"Trial {trial.number} - Fold {fold_idx+1}/{N_SPLITS}")
        
        X_train = X[train_idx]
        y_train = y_binario[train_idx]
        
        X_val = X[val_idx]
        y_val = y_ternario[val_idx]
        
        # Estadísticas del fold
        n_train_clase_0 = (y_train == 0).sum()
        n_train_clase_1 = (y_train == 1).sum()
        
        n_val_total = len(y_val)
        n_val_clase_0 = (y_val == 0).sum()
        n_val_clase_1 = (y_val == 1).sum()
        pct_clase_1 = (n_val_clase_1 / n_val_total * 100) if n_val_total > 0 else 0
        
        logger.info(f"  TRAIN: Clase 0={n_train_clase_0:,}, Clase 1={n_train_clase_1:,}")
        logger.info(f"  VAL: Total={n_val_total:,}, Clase 1 (BAJA+2)={n_val_clase_1:,} ({pct_clase_1:.1f}%)")
        
        fold_stats = {
            'fold': fold_idx + 1,
            'train_clase_0': int(n_train_clase_0),
            'train_clase_1': int(n_train_clase_1),
            'val_total': int(n_val_total),
            'val_clase_1': int(n_val_clase_1),
            'val_clase_1_pct': float(pct_clase_1)
        }
        
        # Entrenar con target_binario
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        # Evaluar con target_ternario
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            valid_sets=[val_set],
            valid_names=['validation'],
            feval=ganancia_evaluator,
            callbacks=[
                lgb.early_stopping(int(50 + 0.05 / learning_rate)),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Evaluar
        y_pred = model.predict(X_val)
        _, ganancia_val, _ = ganancia_evaluator(y_pred, lgb.Dataset(X_val, label=y_val))
        
        ganancias_folds.append(ganancia_val)
        best_iterations.append(model.best_iteration)
        fold_stats['ganancia'] = float(ganancia_val)
        stats_folds.append(fold_stats)
        
        logger.info(f"  Ganancia: {ganancia_val:,.0f}, Best iteration: {model.best_iteration}")
        
        del X_train, y_train, X_val, y_val, model, train_set, val_set, y_pred
        gc.collect()
    
    # Promediar resultados
    ganancia_promedio = np.mean(ganancias_folds)
    ganancia_std = np.std(ganancias_folds)
    best_iteration_promedio = int(np.mean(best_iterations))
    
    trial.set_user_attr('ganancias_folds', [float(g) for g in ganancias_folds])
    trial.set_user_attr('ganancia_std', float(ganancia_std))
    trial.set_user_attr('best_iteration', best_iteration_promedio)
    trial.set_user_attr('best_iterations_folds', best_iterations)
    trial.set_user_attr('stats_folds', stats_folds)
    
    # Feature importance del último modelo
    feature_importance = model.feature_importance() if 'model' in locals() else []
    if len(feature_importance) > 0:
        feature_names = feature_cols
        top_10 = sorted(zip(feature_names, feature_importance), 
                        key=lambda x: x[1], reverse=True)[:10]
        trial.set_user_attr('top_features', [name for name, _ in top_10])
        trial.set_user_attr('top_importance', [float(imp) for _, imp in top_10])
    
    logger.info(f"Trial {trial.number} - Ganancia promedio: {ganancia_promedio:,.0f} ± {ganancia_std:,.0f}")
    logger.info(f"Best iterations: {best_iterations}, Promedio: {best_iteration_promedio}")
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial.number} - RESUMEN:")
    logger.info(f"{'='*60}")
    for stats in stats_folds:
        logger.info(f"Fold {stats['fold']}: "
                   f"Clase 1={stats['val_clase_1']:,} ({stats['val_clase_1_pct']:.1f}%) | "
                   f"Ganancia={stats['ganancia']:,.0f}")
    logger.info(f"{'='*60}\n")
    
    guardar_iteracion(trial, ganancia_promedio)
    
    # Limpiar memoria
    del data, X, y_binario, y_ternario
    gc.collect()
    
    return ganancia_promedio

def guardar_iteracion(trial, ganancia, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    archivo = f"resultados/{archivo_base}_iteraciones.json"
    
    # Crear directorio si no existe
    os.makedirs('resultados', exist_ok=True)
    
    # Datos de esta iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'value': float(ganancia),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE',
'configuracion': {
            'semillas': SEMILLAS,
            'periodos_clase_1': PERIODOS_CLASE_1,
            'periodos_clase_0': PERIODOS_CLASE_0,
            'n_splits': N_SPLITS,
            'undersampling_ratio': UNDERSAMPLING_RATIO
        }
    }
    
    # Cargar datos existentes si el archivo ya existe
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []
    
    # Agregar nueva iteración
    datos_existentes.append(iteracion_data)
    
    # Guardar todas las iteraciones
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
    
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f} - Parámetros: {trial.params}")

def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLAS[0] if isinstance(SEMILLAS, list) else SEMILLAS
    
    bucket_path = os.path.expanduser(BUCKET_NAME)
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(bucket_path, "optuna_db")
    os.makedirs(path_db, exist_ok=True)

    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
 
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"Base de datos encontrada: {db_file}")
        logger.info(f"Cargando estudio existente: {study_name}")
  
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)
  
            logger.info(f"Estudio cargado exitosamente")
            logger.info(f"Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"No se pudo cargar el estudio: {e}")
            logger.info(f"Creando nuevo estudio...")
    else:
        logger.info(f"No se encontró base de datos previa")
        logger.info(f"Creando nueva base de datos: {db_file}")
  
    # Crear nuevo estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=STUDY_NAME,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=SEMILLAS[0]),
        load_if_exists=True
    )
  
    logger.info(f"Nuevo estudio creado: {study_name}")
    logger.info(f"Storage: {storage}")
  
    return study


def optimizar(conn, tabla: str, study_name: str = None, n_trials=100) -> optuna.Study:
    """Ejecuta la optimización con 5-fold CV."""
    study_name = STUDY_NAME
    
    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Cross Validation: {N_SPLITS}-Fold estándar")
    logger.info(f"Períodos clase 1: {len(PERIODOS_CLASE_1)} meses ({PERIODOS_CLASE_1[0]} a {PERIODOS_CLASE_1[-1]})")
    logger.info(f"Períodos clase 0: {len(PERIODOS_CLASE_0)} meses ({PERIODOS_CLASE_0[0]} a {PERIODOS_CLASE_0[-1]})")
    logger.info(f"Undersampling clase 0: {UNDERSAMPLING_RATIO * 100}%")

    study = crear_o_cargar_estudio(study_name, SEMILLAS[0])

    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"Retomando desde trial {trials_previos}")
        logger.info(f"Trials a ejecutar: {trials_a_ejecutar}")
    else:
        logger.info(f"Nueva optimización: {n_trials} trials")

    if trials_a_ejecutar > 0:
        study.optimize(
            lambda trial: objetivo_ganancia(trial, conn, tabla),
            n_trials=trials_a_ejecutar
        )
        logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"Ya se completaron {n_trials} trials")
    
    return study

def evaluar_en_test(conn, tabla: str, study: optuna.Study, mes_test: str) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en test.
    Usa los mismos períodos de entrenamiento que en CV.
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {mes_test}")

    mejores_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']

    periodos_clase1_str = ','.join(map(str, PERIODOS_CLASE_1))
    periodos_clase0_str = ','.join(map(str, PERIODOS_CLASE_0))
    
    logger.info(f"Entrenando con los mismos períodos de CV:")
    logger.info(f"  Clase 1: {len(PERIODOS_CLASE_1)} meses")
    logger.info(f"  Clase 0: {len(PERIODOS_CLASE_0)} meses (undersampling {UNDERSAMPLING_RATIO*100}%)")
    
    # Query con los mismos criterios que en CV
    query_train_completo = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_clase0_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {SEMILLAS[0]})
        ),
        clase_1_completa AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_clase1_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    periodos_test_str = ','.join(map(str, mes_test))
    query_test = f"SELECT * FROM {tabla} WHERE foto_mes IN ({periodos_test_str})"

    train_data = conn.execute(query_train_completo).fetchnumpy()
    test_data = conn.execute(query_test).fetchnumpy()
    
    # Log de tamaños
    n_clase_0 = (train_data['target_binario'] == 0).sum()
    n_clase_1 = (train_data['target_binario'] == 1).sum()
    
    logger.info(f"Train completo: {len(train_data['target_binario']):,} registros")
    logger.info(f"  Clase 0: {n_clase_0:,} | Clase 1: {n_clase_1:,} | Ratio: {n_clase_1/n_clase_0:.3f}:1")
    logger.info(f"Test: {len(test_data['target_binario']):,} registros")
    
    # Preparar features
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario','foto_mes']]
    
    X_train_completo = np.column_stack([train_data[col] for col in feature_cols])
    y_train_completo = train_data['target_binario']
    
    X_test = np.column_stack([test_data[col] for col in feature_cols])
    y_test = test_data['target_ternario']

    models = []
    y_pred_futuro = []
    
    # Entrenar con múltiples semillas
    for i, semilla in enumerate(SEMILLAS):       
        logger.info(f"Entrenando modelo {i+1}/{len(SEMILLAS)} con semilla {semilla}")
        
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'verbose': -1,
            'is_unbalance': True,
            'bagging_freq': 1,
            'n_jobs': -1,
            'seed': semilla,
            'num_leaves': mejores_params['num_leaves'],
            'learning_rate': mejores_params['learning_rate'],
            'min_data_in_leaf': mejores_params['min_data_in_leaf'],
            'feature_fraction': mejores_params['feature_fraction'],
            'bagging_fraction': mejores_params['bagging_fraction'],
            'reg_alpha': mejores_params['reg_alpha'],
            'reg_lambda': mejores_params['reg_lambda'],
            'max_depth': mejores_params['max_depth'] 
        }

        train_set = lgb.Dataset(
            X_train_completo,
            label=y_train_completo,
            feature_name=feature_cols
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )
    
        y_pred = model.predict(X_test)
        _, ganancia_test_semilla, _ = ganancia_evaluator(y_pred, lgb.Dataset(X_test, label=y_test))
        logger.info(f"  Ganancia con semilla {semilla}: {ganancia_test_semilla:,.0f}")
        
        models.append(model)
        y_pred_futuro.append(y_pred)
        
        del train_set
        gc.collect()
    
    # Promedio de predicciones
    pred_matrix = np.column_stack(y_pred_futuro)
    y_pred_promedio = pred_matrix.mean(axis=1) 
    
    # Calcular ganancia
    _, ganancia_test, _ = ganancia_evaluator(y_pred_promedio, lgb.Dataset(X_test, label=y_test))
    
    # Estadísticas
    df_eval = pl.DataFrame({
        'y_true': y_test,
        'y_pred_proba': y_pred_promedio
    })
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col('y_true') == 1)
          .then(GANANCIA_ACIERTO)
          .otherwise(-COSTO_ESTIMULO)
          .cast(pl.Int64)
          .alias('ganancia_individual')
    ])
    df_ordenado = df_ordenado.with_columns([
        pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
    ])
    
    idx_max = df_ordenado.select(pl.col('ganancia_acumulada').arg_max()).item()
    
    predicciones_positivas = idx_max + 1
    total_predicciones = len(y_test)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'parametros_usados': mejores_params,
        'best_iteration': int(best_iteration),
        'periodos_clase_1': PERIODOS_CLASE_1,
        'periodos_clase_0': PERIODOS_CLASE_0,
        'undersampling_ratio': UNDERSAMPLING_RATIO
    }
    
    logger.info(f"=== RESULTADOS FINALES ===")
    logger.info(f"Ganancia en test: {ganancia_test:,.0f}")
    logger.info(f"Enviarías estímulo a: {predicciones_positivas:,} clientes ({porcentaje_positivas:.2f}%)")
    
    # Limpiar memoria
    del X_train_completo, y_train_completo, X_test, y_test, train_data, test_data
    del models, y_pred_futuro, pred_matrix, y_pred_promedio
    gc.collect()
    
    return resultados

def guardar_resultados_test(resultados_test, mes_test, archivo_base=None):
    """
    Guarda resultados de test en JSON.
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    bucket_path = os.path.expanduser(BUCKET_NAME)

    path_resultados = os.path.join(bucket_path, "resultados_test")
    os.makedirs(path_resultados, exist_ok=True)

    archivo_json = os.path.join(path_resultados, f"{archivo_base}_test_results.json")
    
    # Cargar resultados existentes si el archivo ya existe
    if os.path.exists(archivo_json):
        with open(archivo_json, 'r') as f:
            try:
                datos_existentes = json.load(f)
                # Asegurarse de que sea una lista
                if not isinstance(datos_existentes, list):
                    datos_existentes = [datos_existentes]
            except json.JSONDecodeError:
                logger.warning(f"No se pudo leer {archivo_json}, creando nuevo archivo")
                datos_existentes = []
    else:
        datos_existentes = []

    # Agregar timestamp
    resultados_test['datetime'] = datetime.now().isoformat()
    resultados_test['configuracion'] = {
        'semillas': SEMILLAS,
        'periodos_train': PERIODOS_TRAIN,
        'mes_test': mes_test,
        'undersampling_ratio': UNDERSAMPLING_RATIO
    }
    
    datos_existentes.append(resultados_test)

    # Guardar todos los resultados
    with open(archivo_json, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
    
    logger.info(f"Resultados de test guardados en {archivo_json}")
    logger.info(f"Total de evaluaciones acumuladas: {len(datos_existentes)}")

def generar_time_series_splits(periodos: list, n_splits: int, 
                               strategy: str = "expanding",
                               min_train_size: int = 2,
                               val_size: int = 1,
                               gap: int = 0) -> list:
    splits = []
    total_periods = len(periodos)
    
    available_periods = total_periods - min_train_size - gap - val_size
    if available_periods < n_splits - 1:
        logger.warning(f"No hay suficientes períodos para {n_splits} splits")
        n_splits = available_periods + 1
    
    step = max(1, available_periods // (n_splits - 1)) if n_splits > 1 else 1
    
    for i in range(n_splits):
        if strategy == "expanding":
            train_end_idx = min_train_size + (i * step)
        else:
            train_size = min_train_size
            train_end_idx = min_train_size + (i * step)
            train_start_idx = train_end_idx - train_size
            
        val_start_idx = train_end_idx + gap
        val_end_idx = val_start_idx + val_size
        
        if val_end_idx > total_periods:
            break
            
        if strategy == "expanding":
            train_periods = periodos[:train_end_idx]
        else:
            train_periods = periodos[train_start_idx:train_end_idx]
            
        val_periods = periodos[val_start_idx:val_end_idx]
        
        splits.append((train_periods, val_periods))
        
        logger.info(f"Split {i+1}: Train={train_periods}, Val={val_periods}")
    
    return splits

# Uso
if __name__ == "__main__":
    # Conectar a DuckDB
    conn = duckdb.connect('mi_database.duckdb', read_only=True)
    
    # Optimizar
    study = optimizar(conn, tabla='dataset_competencia', n_trials=100)
    
    # Cerrar conexión
    conn.close()