import optuna
import gc
import lightgbm as lgb
import duckdb
import numpy as np
import logging
import json
import os
import pickle
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
                       if col not in ['target_binario', 'target_ternario','foto_mes']]
        
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
    guardar_iteracion(trial, ganancia_promedio, conn)
    
    return ganancia_promedio

def guardar_iteracion(trial, ganancia, conn_duckdb, archivo_base=None):
    """
    Guarda cada iteración de la optimización en un único archivo JSON.
    Y sincroniza la DB con GCS después de cada trial.
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    os.makedirs("resultados", exist_ok=True)
    archivo = f"resultados/{archivo_base}_iteraciones.json"
    
    # Cargar iteraciones existentes
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                iteraciones = json.load(f)
            except json.JSONDecodeError:
                iteraciones = []
    else:
        iteraciones = []
    
    # Crear registro de esta iteración
    iteracion = {
        'trial_number': trial.number,
        'ganancia': float(ganancia),
        'params': trial.params,
        'datetime': datetime.now().isoformat(),
        'user_attrs': {k: v for k, v in trial.user_attrs.items()}
    }
    
    iteraciones.append(iteracion)
    
    # Guardar todas las iteraciones
    with open(archivo, 'w') as f:
        json.dump(iteraciones, f, indent=2)
    
    logger.info(f"Iteración {trial.number} guardada en {archivo}")
    logger.info(f"Ganancia: {ganancia:,.0f} - Parámetros: {trial.params}")
    
    # SINCRONIZAR DB CON GCS
    sincronizar_db_con_gcs(conn_duckdb)

def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente.
    """
    import subprocess
    
    study_name = STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLAS[0] if isinstance(SEMILLAS, list) else SEMILLAS
    
    local_db_dir = os.path.expanduser("~/optuna_db")
    os.makedirs(local_db_dir, exist_ok=True)

    db_file = os.path.join(local_db_dir, f"{study_name}.db")
    gcs_path = f"{BUCKET_NAME}optuna_db/{study_name}.db"
    storage = f"sqlite:///{db_file}"
    
    # DESCARGAR DESDE GCS SI EXISTE (con gsutil en lugar de DuckDB)
    try:
        logger.info(f"Buscando DB en GCS: {gcs_path}")
        
        result = subprocess.run(
            ['gsutil', 'cp', gcs_path, db_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✓ DB descargada desde GCS")
        
    except subprocess.CalledProcessError as e:
        logger.info(f"No hay DB en GCS (normal en primera ejecución)")
    
    # CARGAR O CREAR ESTUDIO
    if os.path.exists(db_file):
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials = len(study.trials)
            logger.info(f"✓ Estudio cargado - {n_trials} trials previos")
            
            if n_trials > 0:
                logger.info(f"✓ Mejor ganancia: {study.best_value:,.0f}")
            
            return study
            
        except Exception as e:
            logger.warning(f"Error al cargar: {e}")
    
    # CREAR NUEVO
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=semilla),
        load_if_exists=True
    )
  
    logger.info(f"✓ Nuevo estudio creado")
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

def evaluar_en_test(conn, tabla: str, study: optuna.Study, mes_test: str, 
                    es_test_2: bool = False) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en test.
    Usa TODOS los períodos con undersampling (estrategia expanding).
    
    Args:
        conn: Conexión a DuckDB
        tabla: Nombre de la tabla
        study: Estudio de Optuna
        mes_test: Período de test
        es_test_2: Si es True, es el último test_2
    
    Returns:
        dict: Resultados de evaluación en test
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {mes_test}")

    mejores_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']

    periodos_train = PERIODOS_TRAIN if es_test_2 else PERIODOS_TRAIN[:-1]
    periodos_train_str = ",".join(str(p) for p in periodos_train)
    
    logger.info(f"Entrenando con {len(periodos_train)} meses disponibles")
    logger.info(f"Períodos: {periodos_train[0]} a {periodos_train[-1]}")
    
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
    resultados_por_semilla = []

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
        
        # Evaluar con ganancia_evaluator
        _, ganancia_test_semilla, _ = ganancia_evaluator(
            y_pred_futuro[i], 
            lgb.Dataset(X_test, label=y_test)
        )
        
        # Obtener threshold y envíos óptimos (guardados por ganancia_evaluator)
        threshold_optimo = ganancia_evaluator.last_threshold
        envios_optimos = ganancia_evaluator.last_envios
    
        resultados_por_semilla.append({
            'semilla': int(SEMILLAS[i]),
            'threshold': float(threshold_optimo),
            'ganancia': float(ganancia_test_semilla),
            'envios': int(envios_optimos),
            'porcentaje_envios': float(envios_optimos / len(y_test) * 100)
        })
        
        logger.info(f"Semilla {SEMILLAS[i]}: Threshold={threshold_optimo:.4f}, "
                   f"Ganancia={ganancia_test_semilla:,.0f}, Envíos={envios_optimos:,}")
        
        del train_set
        gc.collect()
    
    # Promedio de predicciones (ENSEMBLE)
    pred_matrix = np.column_stack(y_pred_futuro)
    y_pred_promedio = pred_matrix.mean(axis=1) 
    
    # Calcular ganancia del ensemble
    _, ganancia_test, _ = ganancia_evaluator(
        y_pred_promedio, 
        lgb.Dataset(X_test, label=y_test)
    )

    # Obtener threshold y envíos del ensemble
    threshold_ensemble = ganancia_evaluator.last_threshold
    envios_ensemble = ganancia_evaluator.last_envios
    
    logger.info(f"=== ENSEMBLE ===")
    logger.info(f"Threshold óptimo: {threshold_ensemble:.4f}")
    logger.info(f"Ganancia máxima: {ganancia_test:,.0f}")
    logger.info(f"Envíos óptimos: {envios_ensemble:,} ({envios_ensemble/len(y_test)*100:.2f}%)")
    
    resultados = {
        'mes_test': mes_test,
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(len(y_test)),
        'predicciones_positivas': int(envios_ensemble),
        'porcentaje_positivas': float(envios_ensemble / len(y_test) * 100),
        'threshold_ensemble': float(threshold_ensemble),
        'parametros_usados': mejores_params,
        'best_iteration': int(best_iteration),
        'periodos_train_usados': periodos_train,
        'undersampling_ratio': UNDERSAMPLING_RATIO,
        'resultados_por_semilla': resultados_por_semilla
    }

    # Limpiar memoria
    del X_train_completo, y_train_completo, X_test, y_test, train_data, test_data
    del models, y_pred_futuro, pred_matrix, y_pred_promedio
    gc.collect()
    
    return resultados

def entrenar_y_guardar_modelos_finales(conn, tabla: str, study: optuna.Study):
    """
    Entrena los modelos finales usando PERIODOS_TRAIN + MES_TEST_2 y los guarda.
    """
    logger.info("=== ENTRENAMIENTO DE MODELOS FINALES ===")
    logger.info(f"Entrenando con PERIODOS_TRAIN + MES_TEST_2")
    
    mejores_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']
    
    # Combinar PERIODOS_TRAIN + MES_TEST_2
    periodos_finales = PERIODOS_TRAIN + MES_TEST_2
    logger.info(f"Períodos totales: {len(periodos_finales)}")
    logger.info(f"Rango: {periodos_finales[0]} a {periodos_finales[-1]}")
    
    periodos_str = ','.join(map(str, periodos_finales))
    
    # Query con undersampling
    query_train_final = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {SEMILLAS[0]})
        ),
        clase_1_completa AS (
            SELECT * FROM {tabla}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    train_data = conn.execute(query_train_final).fetchnumpy()
    
    n_clase_0 = (train_data['target_binario'] == 0).sum()
    n_clase_1 = (train_data['target_binario'] == 1).sum()
    
    logger.info(f"Datos finales (post-undersampling): {len(train_data['target_binario']):,} registros")
    logger.info(f"  Clase 0: {n_clase_0:,} | Clase 1: {n_clase_1:,} | Ratio: {n_clase_1/n_clase_0:.2f}:1")
    
    # Preparar features
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario','foto_mes']]
    
    X_train = np.column_stack([train_data[col] for col in feature_cols])
    y_train = train_data['target_binario']
    
    # TODO EN EL DIRECTORIO LOCAL "modelos_finales/"
    path_modelos = "modelos_finales"
    os.makedirs(path_modelos, exist_ok=True)
    
    models_finales = []
    
    # Entrenar un modelo por cada semilla
    for i, semilla in enumerate(SEMILLAS):
        logger.info(f"Entrenando modelo final {i+1}/{len(SEMILLAS)} con semilla {semilla}")
        
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
            X_train,
            label=y_train,
            feature_name=feature_cols
        )
        
        modelo = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        models_finales.append(modelo)
        
        # Guardar modelo individual
        archivo_modelo = os.path.join(path_modelos, f"{STUDY_NAME}_seed_{semilla}.txt")
        modelo.save_model(archivo_modelo)
        logger.info(f"Modelo guardado: {archivo_modelo}")
        
        del train_set
        gc.collect()
    
    # Guardar información del ensemble
    ensemble_info = {
        'study_name': STUDY_NAME,
        'n_models': len(SEMILLAS),
        'semillas': SEMILLAS,
        'parametros': mejores_params,
        'best_iteration': best_iteration,
        'feature_cols': feature_cols,
        'periodos_entrenamiento': periodos_finales,
        'undersampling_ratio': UNDERSAMPLING_RATIO,
        'datetime': datetime.now().isoformat()
    }
    
    # Guardar metadata en JSON
    archivo_ensemble_info = os.path.join(path_modelos, f"{STUDY_NAME}_ensemble_info.json")
    with open(archivo_ensemble_info, 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    logger.info(f"Información del ensemble guardada: {archivo_ensemble_info}")
    
    # Guardar también un pickle con todos los modelos
    archivo_ensemble_pkl = os.path.join(path_modelos, f"{STUDY_NAME}_ensemble.pkl")
    with open(archivo_ensemble_pkl, 'wb') as f:
        pickle.dump({
            'models': models_finales,
            'info': ensemble_info
        }, f)
    logger.info(f"Ensemble completo guardado: {archivo_ensemble_pkl}")
    
    logger.info("=== MODELOS FINALES GUARDADOS EXITOSAMENTE ===")
    logger.info(f"Total de modelos: {len(models_finales)}")
    logger.info(f"Ubicación: {path_modelos}")
    
    # Limpiar memoria
    del X_train, y_train, train_data, models_finales
    gc.collect()
    
    return ensemble_info

def guardar_resultados_test(resultados_test, mes_test, archivo_base=None):
    """
    Guarda resultados de test en JSON.
    """
    if archivo_base is None:
        archivo_base = STUDY_NAME
    
    # TODO EN EL DIRECTORIO LOCAL "resultados_test/"
    path_resultados = "resultados_test"
    os.makedirs(path_resultados, exist_ok=True)

    archivo_json = os.path.join(path_resultados, f"{archivo_base}_test_results.json")
    
    # Cargar resultados existentes si el archivo ya existe
    if os.path.exists(archivo_json):
        with open(archivo_json, 'r') as f:
            try:
                datos_existentes = json.load(f)
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

def sincronizar_db_con_gcs(conn_duckdb=None):
    """
    Actualiza la DB en GCS usando gsutil.
    """
    import subprocess
    
    local_db_dir = os.path.expanduser("~/optuna_db")
    db_file = os.path.join(local_db_dir, f"{STUDY_NAME}.db")
    
    if not os.path.exists(db_file):
        logger.warning(f"No se encontró DB local: {db_file}")
        return
    
    gcs_path = f"{BUCKET_NAME}optuna_db/{STUDY_NAME}.db"
    
    try:
        result = subprocess.run(
            ['gsutil', 'cp', db_file, gcs_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        file_size = os.path.getsize(db_file)
        logger.info(f"✓ DB actualizada en GCS ({file_size:,} bytes)")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error al sincronizar con gsutil: {e.stderr}")
    except Exception as e:
        logger.warning(f"Error al sincronizar: {e}")

def sincronizar_resultados_con_gcs():
    """
    Sube todos los resultados locales a GCS.
    """
    import subprocess
    
    archivos_a_subir = [
        ('resultados/', f'{BUCKET_NAME}resultados/'),
        ('modelos_finales/', f'{BUCKET_NAME}modelos_finales/'),
        ('resultados_test/', f'{BUCKET_NAME}resultados_test/')
    ]
    
    for local_path, gcs_path in archivos_a_subir:
        if os.path.exists(local_path):
            try:
                result = subprocess.run(
                    ['gsutil', '-m', 'rsync', '-r', local_path, gcs_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"✔ Sincronizado: {local_path} -> {gcs_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Error sincronizando {local_path}: {e.stderr}")
            except Exception as e:
                logger.warning(f"Error: {e}")
        else:
            logger.warning(f"⚠ Path local no existe: {local_path}")