import optuna
import gc
import lightgbm as lgb
import duckdb
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import *

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial, conn, tabla: str, cv_splits: list) -> float:
    """
    Función objetivo con Time Series CV y undersampling.
    """
    # Hiperparámetros a optimizar f 
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
    
    ganancias_folds = []
    best_iterations = []
    stats_folds = []

    # LOOP SOBRE FOLDS
    for fold_idx, (train_periods, val_periods) in enumerate(cv_splits):
        logger.info(f"Trial {trial.number} - Fold {fold_idx+1}/{len(cv_splits)}")
        
        # Query SIMPLE con undersampling (sin separación baja/continua)
        periodos_train_str = ','.join(map(str, train_periods))
        
        query_train = f"""
            WITH clase_0_sample AS (
                SELECT * FROM {tabla}
                WHERE foto_mes IN ({periodos_train_str}) 
                  AND target_binario = 0
                USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {SEMILLAS[0] + fold_idx})
            ),
            clase_1_completa AS (
                SELECT * FROM {tabla}
                WHERE foto_mes IN ({periodos_train_str}) 
                  AND target_binario = 1
            )
            SELECT * FROM clase_0_sample
            UNION ALL
            SELECT * FROM clase_1_completa
        """
        
        periodos_val_str = ','.join(map(str, val_periods))
        query_val = f"SELECT * FROM {tabla} WHERE foto_mes IN ({periodos_val_str})"
        
        # Obtener datos
        train_data = conn.execute(query_train).fetchnumpy()
        val_data = conn.execute(query_val).fetchnumpy()

                # ESTADÍSTICAS DETALLADAS
        n_train_clase_0 = (train_data['target_binario'] == 0).sum()
        n_train_clase_1 = (train_data['target_binario'] == 1).sum()
        
        n_val_total = len(val_data['target_ternario'])
        n_val_continua = (val_data['target_ternario'] == 0).sum()
        n_val_baja1 = (val_data['target_ternario'] == 2).sum()
        n_val_baja2 = (val_data['target_ternario'] == 1).sum()
        
        pct_baja2 = (n_val_baja2 / n_val_total * 100) if n_val_total > 0 else 0
        
        logger.info(f"Trial {trial.number} - Fold {fold_idx+1} - TRAIN:")
        logger.info(f"  Clase 0: {n_train_clase_0:,} | Clase 1: {n_train_clase_1:,}")
        logger.info(f"Trial {trial.number} - Fold {fold_idx+1} - VALIDACIÓN:")
        logger.info(f"  Total: {n_val_total:,}")
        logger.info(f"  CONTINUA (0): {n_val_continua:,} ({n_val_continua/n_val_total*100:.1f}%)")
        logger.info(f"  BAJA+1 (2): {n_val_baja1:,} ({n_val_baja1/n_val_total*100:.1f}%)")
        logger.info(f"  BAJA+2 (1): {n_val_baja2:,} ({pct_baja2:.1f}%) ← OBJETIVO")
        
        fold_stats = {
            'fold': fold_idx + 1,
            'val_periods': val_periods,
            'val_baja2': int(n_val_baja2),
            'val_baja2_pct': float(pct_baja2)
        }
        
        logger.info(f"Trial {trial.number} - Fold {fold_idx+1} - Train: Clase 0={(train_data['target_binario']==0).sum():,}, Clase 1={(train_data['target_binario']==1).sum():,}")
        
        # Preparar features
        feature_cols = [col for col in train_data.keys() 
                       if col not in ['target_binario', 'target_ternario', 'foto_mes']]
        
        X_train = np.column_stack([train_data[col] for col in feature_cols])
        y_train = train_data['target_binario']
        
        X_val = np.column_stack([val_data[col] for col in feature_cols])
        y_val = val_data['target_ternario']
        
        # Entrenar
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
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
        
        logger.info(f"Trial {trial.number} - Fold {fold_idx+1} - Ganancia: {ganancia_val:,.0f}")
        
        del X_train, y_train, X_val, y_val, train_data, val_data, model, train_set, val_set, y_pred
        gc.collect()
    
    # Promediar
    ganancia_promedio = np.mean(ganancias_folds)
    ganancia_std = np.std(ganancias_folds)
    best_iteration_promedio = int(np.mean(best_iterations))
    
    trial.set_user_attr('ganancias_folds', [float(g) for g in ganancias_folds])
    trial.set_user_attr('ganancia_std', float(ganancia_std))
    trial.set_user_attr('best_iteration', best_iteration_promedio)
    trial.set_user_attr('best_iterations_folds', best_iterations)
    trial.set_user_attr('stats_folds', stats_folds)
    
    # Guardar feature importance del último fold
    if 'model' in locals():
        feature_importance = model.feature_importance()
        feature_names = model.feature_name()
        top_10 = sorted(zip(feature_names, feature_importance), 
                        key=lambda x: x[1], reverse=True)[:10]
        trial.set_user_attr('top_features', [name for name, _ in top_10])
        trial.set_user_attr('top_importance', [float(imp) for _, imp in top_10])
    
    logger.info(f"Trial {trial.number} - Ganancia promedio: {ganancia_promedio:,.0f} ± {ganancia_std:,.0f}")
    logger.info(f"\n{'='*60}")
    logger.info(f"Trial {trial.number} - RESUMEN:")
    logger.info(f"{'='*60}")
    for stats in stats_folds:
        logger.info(f"Fold {stats['fold']}: Val={stats['val_periods']} | "
                   f"BAJA+2={stats['val_baja2']:,} ({stats['val_baja2_pct']:.1f}%) | "
                   f"Ganancia={stats['ganancia']:,.0f}")
    logger.info(f"{'='*60}\n")
    guardar_iteracion(trial, ganancia_promedio)
    
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
            'periodos_train': PERIODOS_TRAIN,
            'n_splits': N_SPLITS,
            'cv_strategy': CV_STRATEGY,
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
    study_name = STUDY_NAME
    
    # Generar períodos de optimización (todos menos test)
    todos_periodos = PERIODOS_TRAIN
    
    # Generar splits de CV
    cv_splits = generar_time_series_splits(
        periodos=todos_periodos,
        n_splits=N_SPLITS,
        strategy=CV_STRATEGY,
        min_train_size=MIN_TRAIN_SIZE,
        val_size=VALIDATION_SIZE,
        gap=GAP
    )

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Time Series CV: {len(cv_splits)} splits, estrategia={CV_STRATEGY}")
    logger.info(f"Undersampling: {UNDERSAMPLING_RATIO * 100}%")
    logger.info(f"Períodos disponibles: {todos_periodos}")

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
            lambda trial: objetivo_ganancia(trial, conn, tabla, cv_splits),
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
    Usa TODOS los períodos con undersampling (estrategia expanding).
    
    Args:
        conn: Conexión a DuckDB
        tabla: Nombre de la tabla
        study: Estudio de Optuna
        mes_test: Período de test
    
    Returns:
        dict: Resultados de evaluación en test
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {mes_test}")

    mejores_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']

    # Usar TODOS los períodos (expanding strategy)
    periodos_train_str = ','.join(map(str, PERIODOS_TRAIN))
    
    logger.info(f"Entrenando con TODOS los {len(PERIODOS_TRAIN)} meses disponibles")
    logger.info(f"Períodos: {PERIODOS_TRAIN[0]} a {PERIODOS_TRAIN[-1]}")
    
    # Query CON UNDERSAMPLING
    query_train_completo = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_train_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {SEMILLAS[0]})
        ),
        clase_1_completa AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_train_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    periodos_test_str = ','.join(map(str, mes_test))
    query_test = f"SELECT * FROM {tabla} WHERE foto_mes in ({periodos_test_str})"

    # Obtener datos
    train_data = conn.execute(query_train_completo).fetchnumpy()
    test_data = conn.execute(query_test).fetchnumpy()
    
    # Log de tamaños
    n_clase_0 = (train_data['target_binario'] == 0).sum()
    n_clase_1 = (train_data['target_binario'] == 1).sum()
    
    logger.info(f"Train completo (post-undersampling): {len(train_data['target_binario']):,} registros")
    logger.info(f"  Clase 0: {n_clase_0:,} | Clase 1: {n_clase_1:,} | Ratio: {n_clase_1/n_clase_0:.2f}:1")
    logger.info(f"Test: {len(test_data['target_binario']):,} registros")
    
    # Preparar features y target
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario','foto_mes']]
    
    X_train_completo = np.column_stack([train_data[col] for col in feature_cols])
    y_train_completo = train_data['target_binario']
    
    X_test = np.column_stack([test_data[col] for col in feature_cols])
    y_test = test_data['target_ternario']

    models = [0] * len(SEMILLAS)
    y_pred_futuro = [0] * len(SEMILLAS)
    
    # Entrenar con mejores parámetros
    for i in range(len(SEMILLAS)):       
        logger.info(f"Entrenando modelo {i+1}/{len(SEMILLAS)} con semilla {SEMILLAS[i]}")
        
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
            'seed': SEMILLAS[i],

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

        models[i] = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )
    
        # Predecir en test
        y_pred_futuro[i] = models[i].predict(X_test)
        _, ganancia_test_semilla, _ = ganancia_evaluator(y_pred_futuro[i], lgb.Dataset(X_test, label=y_test))
        logger.info(f"Ganancia con semilla {SEMILLAS[i]}: {ganancia_test_semilla:,.0f}")
        
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
        'periodos_train_usados': PERIODOS_TRAIN,  # Todos los períodos
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