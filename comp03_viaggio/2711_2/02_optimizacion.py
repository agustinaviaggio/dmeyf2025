import duckdb
import logging
from datetime import datetime
import os
import json
import subprocess

from src.features import create_sql_table_from_parquet_csv
from src.optimization_duck import *
from src.config import *

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Iniciando programa de optimización con log fechado")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH_OPT: {DATA_PATH_OPT}")
logger.info(f"SEMILLAS: {SEMILLAS}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")

### Cargar features seleccionadas desde GCS ###
def cargar_features_seleccionadas(umbral='90pct'):
    """
    Descarga y carga el JSON con las features seleccionadas desde GCS.
    
    Args:
        umbral: Umbral de frecuencia a usar ('union','100pct', '90pct', '80pct', '75pct', '50pct')
    
    Returns:
        lista de features seleccionadas
    """
    logger.info(f"=== CARGANDO FEATURES SELECCIONADAS (umbral: {umbral}) ===")
    
    # Construir patrón de búsqueda según el umbral
    if umbral == 'union':
        # Para union, el archivo tiene otro formato
        gcs_pattern = f"{BUCKET_NAME}resultados/union_features_{STUDY_NAME}_*.json"
    else:
        gcs_pattern = f"{BUCKET_NAME}resultados/features_{umbral}_{STUDY_NAME}_*.json"
    
    result = subprocess.run(
        ['gsutil', 'ls', gcs_pattern],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"No se encontraron archivos con patrón: {gcs_pattern}")
        logger.error(f"Error: {result.stderr}")
        raise FileNotFoundError(f"No se encontró archivo de features con umbral {umbral}")
    
    # Tomar el más reciente (último en la lista)
    archivos = result.stdout.strip().split('\n')
    if not archivos or archivos[0] == '':
        raise FileNotFoundError(f"No se encontró archivo de features con umbral {umbral}")
    
    archivo_mas_reciente = sorted(archivos)[-1]
    logger.info(f"Archivo más reciente encontrado: {archivo_mas_reciente}")
    
    # Descargar a archivo temporal
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        result = subprocess.run(
            ['gsutil', 'cp', archivo_mas_reciente, tmp_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise Exception(f"Error descargando archivo: {result.stderr}")
        
        # Leer JSON
        with open(tmp_path, 'r') as f:
            data = json.load(f)
        
        # Extraer features según el tipo de archivo
        if umbral == 'union':
            features = data['union_features']['lista_completa']
            total = data['union_features']['total_features_en_union']
            logger.info(f"✓ Features cargadas exitosamente (UNIÓN COMPLETA)")
            logger.info(f"  Total de features: {total}")
        else:
            features = data['features']
            total = data['total']
            logger.info(f"✓ Features cargadas exitosamente")
            logger.info(f"  Total de features: {total}")
            logger.info(f"  Umbral: {data['umbral']}")
        
        logger.info(f"  Archivo: {archivo_mas_reciente.split('/')[-1]}")
        
        return features
        
    finally:
        # Limpiar archivo temporal
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML y features seleccionadas."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")

    conn = None
    try:
        # 0. CARGAR FEATURES SELECCIONADAS
        UMBRAL_FEATURES = 'union'
        
        try:
            features_seleccionadas = cargar_features_seleccionadas(umbral=UMBRAL_FEATURES)
            logger.info(f"Se usarán {len(features_seleccionadas)} features para la optimización")
        except FileNotFoundError as e:
            logger.warning(f"No se encontró archivo de features: {e}")
            logger.warning("Se procederá con TODAS las features disponibles")
            features_seleccionadas = None
        
        # 1. Configurar DuckDB y GCS
        conn = duckdb.connect(database=':memory:')
        conn.execute("SET temp_directory='/tmp'")
        
        # Configurar GCS
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
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
        
        if features_seleccionadas is not None:
            logger.info("=== CARGANDO DATASET CON FEATURES SELECCIONADAS ===")
            columnas_necesarias = ['target_binario', 'target_ternario', 'foto_mes']
            columnas_str = ', '.join(features_seleccionadas + columnas_necesarias)
            
            conn.execute(f"""
                CREATE TABLE {SQL_TABLE_NAME} AS 
                SELECT {columnas_str}
                FROM read_parquet('{DATA_PATH_OPT}')
            """)
            logger.info(f"✓ Dataset cargado: {len(features_seleccionadas)} features")
        else:
            conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_OPT, SQL_TABLE_NAME)
        
        # 4. Ejecutar optimización
        study = optimizar(conn, SQL_TABLE_NAME, n_trials=150)
    
        # 5. Análisis adicional
        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        trials_completos = [t for t in study.trials if t.value is not None]
        trials_ordenados = sorted(trials_completos, key=lambda t: t.value, reverse=True)[:5]

        logger.info("Top 5 mejores trials:")
        for trial in trials_ordenados:
            logger.info(f"  Trial {trial.number}: {trial.value:,.0f}")

        # Análisis de feature importance del mejor trial
        logger.info("=== FEATURE IMPORTANCE DEL MEJOR TRIAL ===")
        best_trial = study.best_trial
        top_features = best_trial.user_attrs.get('top_features', [])
        top_importance = best_trial.user_attrs.get('top_importance', [])

        logger.info("Top 10 features más importantes:")
        for name, importance in zip(top_features, top_importance):
            logger.info(f"  {name}: {importance:,.0f}")

        logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

        '''# 6. Evaluación en TEST 1
        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST 1 ===")
        resultados_test = evaluar_en_test(
            conn, 
            SQL_TABLE_NAME, 
            study, 
            MES_TEST_1,
            es_test_2=False
        )
        guardar_resultados_test(resultados_test, MES_TEST_1)
        logger.info("=== RESUMEN DE EVALUACIÓN EN TEST 1 ===")
        logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    
        # 7. Evaluación en TEST 2
        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST 2 ===")
        resultados_test = evaluar_en_test(
            conn, 
            SQL_TABLE_NAME, 
            study, 
            MES_TEST_2,
            es_test_2=True
        )
        guardar_resultados_test(resultados_test, MES_TEST_2)
        logger.info("=== RESUMEN DE EVALUACIÓN EN TEST 2 ===")
        logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
        '''
        # 8. ENTRENAR Y GUARDAR MODELOS FINALES (con TRAIN + TEST_2)
        logger.info("=== ENTRENAMIENTO FINAL CON TRAIN + TEST_2 ===")
        ensemble_info = entrenar_y_guardar_modelos_finales(conn, SQL_TABLE_NAME, study)
        
        logger.info("=== PIPELINE COMPLETADO ===")
        logger.info(f"Modelos finales guardados: {ensemble_info['study_name']}")
        logger.info(f"Ubicación: {os.path.expanduser(BUCKET_NAME)}/modelos_finales/")

        # 9. SINCRONIZAR BASE DE DATOS CON GCS
        logger.info("=== SINCRONIZANDO BASE DE DATOS CON GCS ===")
        sincronizar_db_con_gcs(conn)

        logger.info("=== SINCRONIZANDO RESULTADOS Y MODELOS CON GCS ===")
        sincronizar_resultados_con_gcs()
        
        # 10. Guardar metadata de features usadas
        if features_seleccionadas is not None:
            logger.info("=== GUARDANDO METADATA DE FEATURES USADAS ===")
            metadata = {
                'study_name': STUDY_NAME,
                'fecha_optimizacion': fecha,
                'umbral_features': UMBRAL_FEATURES,
                'n_features_usadas': len(features_seleccionadas),
                'features': features_seleccionadas
            }
            
            metadata_file = f"metadata_features_{STUDY_NAME}_{fecha}.json"
            metadata_path = os.path.join("resultados", metadata_file)
            os.makedirs("resultados", exist_ok=True)
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Subir a GCS
            gcs_path = f"{BUCKET_NAME}resultados/{metadata_file}"
            subprocess.run(['gsutil', 'cp', metadata_path, gcs_path])
            logger.info(f"✓ Metadata guardada en: {gcs_path}")

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()