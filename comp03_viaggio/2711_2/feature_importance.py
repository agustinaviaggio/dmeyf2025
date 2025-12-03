"""
Script de análisis de feature importance.
Analiza los top 5 trials de Optuna con 25 semillas cada uno (125 modelos totales).
Extrae la unión de los top 600 features de cada modelo.
"""

import duckdb
import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
import logging
import json
import subprocess
import os
from datetime import datetime
from collections import Counter
import tempfile

# Configuración
BUCKET_NAME = "gs://sra_electron_bukito3/"
STUDY_NAME = "2511_2"
SQL_TABLE_NAME = "dataset_competencia"
DATA_PATH = "gs://sra_electron_bukito3/datasets/competencia_03_FE_v4.parquet"

# Períodos de entrenamiento (todos los meses del conf.yaml)
PERIODOS_TRAIN = [201901, 201902, 201903, 201904, 201906,
                  201907, 201908, 201909, 201911, 201912,
                  202001, 202002, 202003, 202004, 202005, 202007,
                  202008, 202009, 202010, 202011, 202012,
                  202101, 202102, 202103, 202104, 202105, 202106]

# Semillas del conf.yaml
SEMILLAS = [600011, 600043, 600053, 600071, 600073,
            600091, 600107, 600109, 600113, 600137,
            600169, 600179, 600191, 600193, 600197,
            600209, 600211, 600221, 600227, 600253,
            600257, 600259, 600263, 600281, 600293]

UNDERSAMPLING_RATIO = 0.075
N_SEMILLAS = len(SEMILLAS)
TOP_N_FEATURES = 600

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def configurar_gcs(conn):
    """Configura autenticación para GCS."""
    from google.auth import default
    from google.auth.transport.requests import Request
    
    credentials, _ = default()
    credentials.refresh(Request())
    token = credentials.token
    
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    conn.execute(f"""
        CREATE SECRET (
            TYPE GCS,
            PROVIDER config,
            BEARER_TOKEN '{token}'
        )
    """)
    logger.info("✓ GCS configurado")

def cargar_dataset(conn):
    """Carga el dataset."""
    logger.info(f"Cargando dataset desde {DATA_PATH}...")
    
    conn.execute(f"""
        CREATE TABLE {SQL_TABLE_NAME} AS 
        SELECT * FROM read_parquet('{DATA_PATH}')
    """)
    
    n_rows = conn.execute(f"SELECT COUNT(*) FROM {SQL_TABLE_NAME}").fetchone()[0]
    logger.info(f"✓ Dataset cargado: {n_rows:,} filas")

def descargar_db_optuna():
    """Descarga la base de datos de Optuna desde GCS."""
    # Usar directorio temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as tmp:
        db_file = tmp.name
    
    gcs_path = f"{BUCKET_NAME}optuna_db/{STUDY_NAME}.db"
    
    logger.info(f"Descargando DB de Optuna desde {gcs_path}...")
    
    result = subprocess.run(
        ['gsutil', 'cp', gcs_path, db_file],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info(f"✓ DB descargada a temporal: {db_file}")
        return db_file
    else:
        raise Exception(f"Error descargando DB: {result.stderr}")

def cargar_estudio_optuna(db_file):
    """Carga el estudio de Optuna."""
    storage = f"sqlite:///{db_file}"
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    logger.info(f"✓ Estudio cargado: {len(study.trials)} trials")
    return study

def entrenar_modelo_con_params(conn, params, semilla, trial_num, semilla_idx, contador_modelo, total_modelos):
    """Entrena un modelo con parámetros específicos."""
    logger.info(f"  [{semilla_idx}/{N_SEMILLAS}] Trial {trial_num}, Semilla {semilla} (Modelo {contador_modelo}/{total_modelos})")
    
    # Query de entrenamiento
    periodos_str = ','.join(map(str, PERIODOS_TRAIN))
    
    query_train = f"""
        WITH clase_0_sample AS (
            SELECT * FROM {SQL_TABLE_NAME}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 0
            USING SAMPLE {UNDERSAMPLING_RATIO * 100} PERCENT (bernoulli, {semilla})
        ),
        clase_1_completa AS (
            SELECT * FROM {SQL_TABLE_NAME}
            WHERE foto_mes IN ({periodos_str}) 
              AND target_binario = 1
        )
        SELECT * FROM clase_0_sample
        UNION ALL
        SELECT * FROM clase_1_completa
    """
    
    train_data = conn.execute(query_train).fetchnumpy()
    
    # Preparar features
    feature_cols = [col for col in train_data.keys() 
                   if col not in ['target_binario', 'target_ternario', 'foto_mes']]
    
    X_train = np.column_stack([train_data[col] for col in feature_cols])
    y_train = train_data['target_binario']
    
    # Parámetros del modelo
    train_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'max_bin': 31,
        'is_unbalance': True,
        'boost_from_average': True,
        'feature_pre_filter': True,
        'bagging_freq': 1,
        'n_jobs': -1,
        'seed': semilla,
        'verbose': -1,
        **params
    }
    
    # Entrenar
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    
    model = lgb.train(
        train_params,
        train_set,
        num_boost_round=params['best_iteration'],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    del X_train, y_train, train_data, train_set, model
    
    return importance_df

def guardar_json_gcs(datos, nombre_archivo):
    """Guarda JSON directamente a GCS usando archivo temporal."""
    gcs_path = f"{BUCKET_NAME}resultados/{nombre_archivo}"
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json.dump(datos, tmp, indent=2)
        tmp_path = tmp.name
    
    try:
        # Subir a GCS
        result = subprocess.run(
            ['gsutil', 'cp', tmp_path, gcs_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Guardado en GCS: {gcs_path}")
        else:
            logger.error(f"Error subiendo a GCS: {result.stderr}")
            raise Exception(f"Error subiendo a GCS: {result.stderr}")
    finally:
        # Limpiar archivo temporal
        os.unlink(tmp_path)

def main():
    logger.info("="*70)
    logger.info("ANÁLISIS DE FEATURE IMPORTANCE - 125 MODELOS")
    logger.info(f"Top 5 trials × {N_SEMILLAS} semillas = {5 * N_SEMILLAS} modelos")
    logger.info("="*70)
    
    conn = None
    db_file = None
    
    try:
        # 1. Configurar DuckDB y GCS
        conn = duckdb.connect(database=':memory:')
        configurar_gcs(conn)
        
        # 2. Cargar dataset
        cargar_dataset(conn)
        
        # 3. Cargar estudio de Optuna
        db_file = descargar_db_optuna()
        study = cargar_estudio_optuna(db_file)
        
        # 4. Obtener top 5 trials
        trials_completos = [t for t in study.trials if t.value is not None]
        top_5_trials = sorted(trials_completos, key=lambda t: t.value, reverse=True)[:5]
        
        logger.info("\n" + "="*70)
        logger.info("TOP 5 TRIALS")
        logger.info("="*70)
        for i, trial in enumerate(top_5_trials, 1):
            logger.info(f"{i}. Trial {trial.number}: Ganancia = {trial.value:,.0f}")
        
        logger.info(f"\nSemillas: {len(SEMILLAS)} semillas")
        logger.info(f"Períodos de entrenamiento: {len(PERIODOS_TRAIN)} meses")
        logger.info(f"Undersampling: {UNDERSAMPLING_RATIO*100}%")
        
        # 5. Entrenar todos los modelos y recolectar top features
        all_top_features = []  # Lista de sets con top 600 features de cada modelo
        feature_importance_aggregated = Counter()  # Contador de apariciones
        
        contador_modelo = 0
        total_modelos = len(top_5_trials) * N_SEMILLAS
        
        for trial_idx, trial in enumerate(top_5_trials, 1):
            logger.info("\n" + "="*70)
            logger.info(f"TRIAL {trial.number} ({trial_idx}/5) - Ganancia: {trial.value:,.0f}")
            logger.info("="*70)
            
            params = trial.params.copy()
            params['best_iteration'] = trial.user_attrs.get('best_iteration', 100)
            
            for semilla_idx, semilla in enumerate(SEMILLAS, 1):
                contador_modelo += 1
                
                # Entrenar modelo
                importance_df = entrenar_modelo_con_params(
                    conn, params, semilla, 
                    trial.number, semilla_idx, contador_modelo, total_modelos
                )
                
                # Extraer top N features
                top_features = set(importance_df.head(TOP_N_FEATURES)['feature'].tolist())
                all_top_features.append(top_features)
                
                # Agregar al contador de frecuencias
                for feature in top_features:
                    feature_importance_aggregated[feature] += 1
                
                if contador_modelo % 5 == 0:
                    logger.info(f"    ✓ Progreso: {contador_modelo}/{total_modelos} modelos completados")
        
        # 6. Calcular unión de todos los top features
        logger.info("\n" + "="*70)
        logger.info("CALCULANDO UNIÓN DE FEATURES")
        logger.info("="*70)
        
        union_features = set.union(*all_top_features)
        
        logger.info(f"Total de features en la unión: {len(union_features)}")
        logger.info(f"Modelos entrenados: {total_modelos}")
        
        # 7. Análisis de frecuencias
        features_ordenadas_por_frecuencia = feature_importance_aggregated.most_common()
        
        # Estadísticas de frecuencias
        features_en_todos = [f for f, count in features_ordenadas_por_frecuencia if count == total_modelos]
        features_en_90pct = [f for f, count in features_ordenadas_por_frecuencia if count >= total_modelos * 0.9]
        features_en_80pct = [f for f, count in features_ordenadas_por_frecuencia if count >= total_modelos * 0.8]
        features_en_75pct = [f for f, count in features_ordenadas_por_frecuencia if count >= total_modelos * 0.75]
        features_en_50pct = [f for f, count in features_ordenadas_por_frecuencia if count >= total_modelos * 0.5]
        
        logger.info(f"\nFrecuencias de aparición:")
        logger.info(f"  En todos los {total_modelos} modelos: {len(features_en_todos)}")
        logger.info(f"  En ≥90% modelos ({int(total_modelos*0.9)}+): {len(features_en_90pct)}")
        logger.info(f"  En ≥80% modelos ({int(total_modelos*0.8)}+): {len(features_en_80pct)}")
        logger.info(f"  En ≥75% modelos ({int(total_modelos*0.75)}+): {len(features_en_75pct)}")
        logger.info(f"  En ≥50% modelos ({int(total_modelos*0.5)}+): {len(features_en_50pct)}")
        
        # 8. Timestamp para todos los archivos
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 9. Guardar resultado completo
        resultado_final = {
            'metadata': {
                'fecha_analisis': datetime.now().isoformat(),
                'study_name': STUDY_NAME,
                'n_trials': len(top_5_trials),
                'n_semillas': N_SEMILLAS,
                'semillas': SEMILLAS,
                'total_modelos': total_modelos,
                'top_n_features_por_modelo': TOP_N_FEATURES,
                'periodos_entrenamiento': PERIODOS_TRAIN,
                'undersampling_ratio': UNDERSAMPLING_RATIO
            },
            'union_features': {
                'total_features_en_union': len(union_features),
                'lista_completa': sorted(list(union_features))
            },
            'frecuencias': {
                'features_en_todos_modelos': sorted(features_en_todos),
                'features_en_90pct': sorted(features_en_90pct),
                'features_en_80pct': sorted(features_en_80pct),
                'features_en_75pct': sorted(features_en_75pct),
                'features_en_50pct': sorted(features_en_50pct),
                'n_en_todos': len(features_en_todos),
                'n_en_90pct': len(features_en_90pct),
                'n_en_80pct': len(features_en_80pct),
                'n_en_75pct': len(features_en_75pct),
                'n_en_50pct': len(features_en_50pct),
                'distribucion_completa': {f: c for f, c in features_ordenadas_por_frecuencia}
            },
            'top_5_trials_info': [
                {
                    'trial_number': trial.number,
                    'ganancia': float(trial.value),
                    'params': trial.params
                }
                for trial in top_5_trials
            ]
        }
        
        archivo_completo = f"union_features_{STUDY_NAME}_top{TOP_N_FEATURES}_{total_modelos}modelos_{timestamp}.json"
        guardar_json_gcs(resultado_final, archivo_completo)
        
        # 10. Guardar listas por umbral de frecuencia
        logger.info("\nGuardando archivos por umbral...")
        for umbral, features in [
            ('100pct', features_en_todos),
            ('90pct', features_en_90pct),
            ('80pct', features_en_80pct),
            ('75pct', features_en_75pct),
            ('50pct', features_en_50pct)
        ]:
            lista_umbral = {
                'umbral': umbral,
                'features': sorted(features),
                'total': len(features),
                'metadata': {
                    'study_name': STUDY_NAME,
                    'fecha': timestamp,
                    'modelos_analizados': total_modelos
                }
            }
            archivo_umbral = f"features_{umbral}_{STUDY_NAME}_{timestamp}.json"
            guardar_json_gcs(lista_umbral, archivo_umbral)
        
        # 11. Resumen final
        logger.info("\n" + "="*70)
        logger.info("RESUMEN FINAL")
        logger.info("="*70)
        logger.info(f"Modelos entrenados: {total_modelos}")
        logger.info(f"Top features por modelo: {TOP_N_FEATURES}")
        logger.info(f"Features en la unión total: {len(union_features)}")
        logger.info(f"\nFeatures por frecuencia:")
        logger.info(f"  100% modelos: {len(features_en_todos)}")
        logger.info(f"  ≥90% modelos: {len(features_en_90pct)}")
        logger.info(f"  ≥80% modelos: {len(features_en_80pct)}")
        logger.info(f"  ≥75% modelos: {len(features_en_75pct)}")
        logger.info(f"  ≥50% modelos: {len(features_en_50pct)}")
        
        logger.info(f"\nArchivos guardados en: {BUCKET_NAME}resultados/")
        
        logger.info(f"\n{'='*70}")
        logger.info("ANÁLISIS COMPLETADO EXITOSAMENTE")
        logger.info(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Error durante el análisis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada")
        
        # Limpiar archivo temporal de Optuna DB
        if db_file and os.path.exists(db_file):
            os.unlink(db_file)
            logger.info("Archivos temporales eliminados")

if __name__ == "__main__":
    main()