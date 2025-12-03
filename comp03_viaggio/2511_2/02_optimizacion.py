import duckdb
import logging
from datetime import datetime
import os

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

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")

    conn = None
    try:
        conn = duckdb.connect(database=':memory:')
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
        # 1. Cargar datos y crear tabla sql
        conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_OPT, SQL_TABLE_NAME)
  
        # 2. Ejecutar optimización
        study = optimizar(conn, SQL_TABLE_NAME, n_trials=150)
    
        # 3. Análisis adicional
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

        # 4. Evaluación en TEST 1
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
    
        # 5. Evaluación en TEST 2
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

        # 6. ENTRENAR Y GUARDAR MODELOS FINALES (con TRAIN + TEST_2)
        logger.info("=== ENTRENAMIENTO FINAL CON TRAIN + TEST_2 ===")
        ensemble_info = entrenar_y_guardar_modelos_finales(conn, SQL_TABLE_NAME, study)
        
        logger.info("=== PIPELINE COMPLETADO ===")
        logger.info(f"Modelos finales guardados: {ensemble_info['study_name']}")
        logger.info(f"Ubicación: {os.path.expanduser(BUCKET_NAME)}/modelos_finales/")

        # 7. SINCRONIZAR BASE DE DATOS CON GCS
        logger.info("=== SINCRONIZANDO BASE DE DATOS CON GCS ===")
        sincronizar_db_con_gcs(conn)

        logger.info("=== SINCRONIZANDO RESULTADOS Y MODELOS CON GCS ===")
        sincronizar_resultados_con_gcs()

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()