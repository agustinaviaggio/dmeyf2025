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

def objetivo_ganancia(trial, conn, tabla: str) -> float:
    """
    Función objetivo que maximiza ganancia en mes de validación.
    Usa DuckDB con fetchnumpy() - sin pandas.
    
    Args:
        trial: trial de optuna
        conn: conexión a DuckDB
        tabla: nombre de la tabla en DuckDB
    
    Returns:
        float: ganancia total
    """
    # Hiperparámetros a optimizar
    num_leaves = trial.suggest_int('num_leaves', 5, 40) 
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 200, 5000) 
    feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.6) 
    bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0) 
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    reg_alpha = trial.suggest_float('reg_alpha', 0.5, 50.0, log=True) 
    reg_lambda = trial.suggest_float('reg_lambda', 0.5, 50.0, log=True) 
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
        'boost_from_average': True,
        'feature_pre_filter': True,
        'bagging_freq': 1,
        'n_jobs': -1,
        'seed': SEMILLAS[0],
        'verbose': -1
    }
    
    # Queries para train y validación
    if isinstance(MESES_TRAIN, list):
        periodos_train_str = ','.join(map(str, MESES_TRAIN))
    else:
        periodos_train_str = f"{MESES_TRAIN}"
    
    query_train = f"SELECT * FROM {tabla} WHERE foto_mes IN ({periodos_train_str})"
    query_val = f"SELECT * FROM {tabla} WHERE foto_mes = {MES_VALIDACION}"

    # Obtener datos como dict de numpy arrays
    train_data = conn.execute(query_train).fetchnumpy()
    val_data = conn.execute(query_val).fetchnumpy()
    
    # Preparar features y target
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario']]
    
    X_train = np.column_stack([train_data[col] for col in feature_cols])
    y_train = train_data['target_binario']
    
    X_val = np.column_stack([val_data[col] for col in feature_cols])
    y_val = val_data['target_ternario']
    
    # Entrenar modelo con función de ganancia personalizada
    train_set = lgb.Dataset(
        X_train, 
        label=y_train,
        feature_name=feature_cols
    )

    val_set = lgb.Dataset( 
    X_val,
    label=y_val,
    reference=train_set
)
    
    model = lgb.train(
        params,
        train_set,
        num_boost_round=5000,
        valid_sets=[val_set],
        valid_names=['validation'],
        feval=ganancia_evaluator,
        callbacks=[
            lgb.early_stopping(int(50 + 0.05 / learning_rate)),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # Predecir y calcular ganancia
    y_pred = model.predict(X_val)
    _, ganancia_val, _ = ganancia_evaluator(y_pred, lgb.Dataset(X_val, label=y_val))
    trial.set_user_attr('best_iteration', model.best_iteration)
    
    # Guardar feature importance
    feature_importance = model.feature_importance()
    feature_names = model.feature_name()
    top_10 = sorted(zip(feature_names, feature_importance), 
                    key=lambda x: x[1], reverse=True)[:10]

    trial.set_user_attr('top_features', [name for name, _ in top_10])
    trial.set_user_attr('top_importance', [float(imp) for _, imp in top_10])

    # Guardar iteración
    guardar_iteracion(trial, ganancia_val)
    
    logger.debug(f"Trial {trial.number}: Ganancia = {ganancia_val:,.0f}")

    # Antes de return ganancia_val
    logger.info(f"Trial {trial.number} - Ganancia: {ganancia_val:,.0f}")
    logger.info(f"Trial {trial.number} - Best iteration: {model.best_iteration}")
    logger.info(f"Trial {trial.number} - Params: {trial.params}")
    logger.info(f"Trial {trial.number} - Predicciones únicas: {len(np.unique(y_pred))}")
    logger.info(f"Trial {trial.number} - Rango predicciones: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    logger.info(f"Trial {trial.number} - Target ternario = 1: {(y_val == 1).sum()} de {len(y_val)}")

    del X_train, y_train, X_val, y_val, train_data, val_data, model, train_set, val_set, y_pred
    gc.collect()
    return ganancia_val

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
            'semilla': SEMILLAS,
            'mes_train': MESES_TRAIN,
            'mes_validacion': MES_VALIDACION
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
    """
    Ejecuta optimización bayesiana usando DuckDB.
    
    Args:
        conn: Conexión a DuckDB
        tabla: Nombre de la tabla
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        n_trials: Número de trials
    
    Returns:
        optuna.Study: Estudio con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MESES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLAS[0]}")

    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLAS[0])

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"Retomando desde trial {trials_previos}")
        logger.info(f"Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"Nueva optimización: {n_trials} trials")

    # Ejecutar optimización
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
    Args:
        conn: Conexión a DuckDB
        tabla: Nombre de la tabla
        study
    
    Returns:
        dict: Resultados de evaluación en test
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {mes_test}")

    mejores_params = study.best_params
    best_iteration = study.best_trial.user_attrs['best_iteration']
    
    # Queries para train completo (train + validación) y test
    if isinstance(MESES_TRAIN, list):
        periodos_train_str = ','.join(map(str, MESES_TRAIN + [MES_VALIDACION]))
    else:
        periodos_train_str = f"{MESES_TRAIN},{MES_VALIDACION}"
    
    query_train_completo = f"SELECT * FROM {tabla} WHERE foto_mes IN ({periodos_train_str})"
    query_test = f"SELECT * FROM {tabla} WHERE foto_mes = {mes_test}"
    
    # Obtener datos con fetchnumpy
    train_data = conn.execute(query_train_completo).fetchnumpy()
    test_data = conn.execute(query_test).fetchnumpy()
    
    # Preparar features y target
    feature_cols = [col for col in train_data.keys() 
                    if col not in ['target_binario', 'target_ternario']]
    
    X_train_completo = np.column_stack([train_data[col] for col in feature_cols])
    y_train_completo = train_data['target_binario']
    
    X_test = np.column_stack([test_data[col] for col in feature_cols])
    y_test = test_data['target_ternario']
    
    logger.info(f"Train completo: {len(y_train_completo)} registros")
    logger.info(f"Test: {len(y_test)} registros")

    #models = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #y_pred_futuro = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    models = [0,0,0,0,0]
    y_pred_futuro = [0,0,0,0,0]
    # Entrenar con mejores parámetros
    for i in range(len(SEMILLAS)):       
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'verbose': 0,
            'boost_from_average': True,
            'feature_pre_filter': True,
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
            callbacks=[lgb.log_evaluation(period=100)]
        )
    
        # Predecir en test
        y_pred_futuro[i] = models[i].predict(X_test)
        logger.info(f"Entrenamiento con semilla {i} de {len(SEMILLAS)} completadas")
        _, ganancia_test_semilla, _ = ganancia_evaluator(y_pred_futuro[i], lgb.Dataset(X_test, label=y_test))
        logger.info(f"Ganancia en test para esta semilla: {ganancia_test_semilla:,.0f}")
    
    pred_matrix = np.column_stack(y_pred_futuro)
    y_pred_promedio = pred_matrix.mean(axis=1) 
    
    # Calcular ganancia usando tu función que no necesita threshold
    _, ganancia_test, _ = ganancia_evaluator(y_pred_promedio, lgb.Dataset(X_test, label=y_test))
    
    # Estadísticas: contar cuántos estarían en el corte óptimo
    # (opcional, si querés saber cuántos enviarías)
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
    
    # Encontrar el índice donde está la ganancia máxima
    idx_max = df_ordenado.select(
        pl.col('ganancia_acumulada').arg_max()
    ).item()
    
    predicciones_positivas = idx_max + 1  # +1 porque es índice 0-based
    total_predicciones = len(y_test)
    porcentaje_positivas = (predicciones_positivas / total_predicciones) * 100
    
    resultados = {
        'ganancia_test': float(ganancia_test),
        'total_predicciones': int(total_predicciones),
        'predicciones_positivas': int(predicciones_positivas),
        'porcentaje_positivas': float(porcentaje_positivas),
        'parametros_usados': mejores_params
    }
    
    logger.info(f"Ganancia en test: {ganancia_test:,.0f}")
    logger.info(f"Enviarías estímulo a: {predicciones_positivas} clientes ({porcentaje_positivas:.2f}%)")
    
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
        'semilla': SEMILLAS[0],
        'meses_train': MESES_TRAIN + [MES_VALIDACION],
        'mes_test': mes_test
    }
    
    datos_existentes.append(resultados_test)

    # Guardar todos los resultados
    with open(archivo_json, 'w') as f:
        json.dump(datos_existentes, f, indent=2)
    
    logger.info(f"Resultados de test guardados en {archivo_json}")
    logger.info(f"Total de evaluaciones acumuladas: {len(datos_existentes)}")


# Uso
if __name__ == "__main__":
    # Conectar a DuckDB
    conn = duckdb.connect('mi_database.duckdb', read_only=True)
    
    # Optimizar
    study = optimizar(conn, tabla='dataset_competencia', n_trials=100)
    
    # Cerrar conexión
    conn.close()