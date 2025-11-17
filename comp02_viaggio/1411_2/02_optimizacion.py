import logging
from datetime import datetime
import os

from src.features import create_sql_table_from_parquet, target_binario, target_ternario, generar_targets
from src.optimization_duck import optimizar, evaluar_en_test, guardar_resultados_test
from src.config import *
from src.best_params import cargar_mejores_hiperparametros

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
logger.info(f"MESES_TRAIN_BAJA: {MESES_TRAIN_BAJA}")
logger.info(f"MESES_TRAIN_CONTINUA: {MESES_TRAIN_CONTINUA}")
logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
logger.info(f"GANANCIA_ACIERTO: {GANANCIA_ACIERTO}")
logger.info(f"COSTO_ESTIMULO: {COSTO_ESTIMULO}")

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")

    conn = None # Inicializamos la conexión a None
    try:  
        # 1. Cargar datos y crear tabla sql
        conn = create_sql_table_from_parquet(DATA_PATH_OPT, SQL_TABLE_NAME)
  
        # 2. Ejecutar optimización
        study = optimizar(conn, SQL_TABLE_NAME, n_trials=50)
    
        # 5. Análisis adicional
        logger.info("=== ANÁLISIS DE RESULTADOS ===")
        # Filtrar solo trials que completaron exitosamente
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

        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST 1===")
        
        # Evaluar en test
        resultados_test = evaluar_en_test(conn, SQL_TABLE_NAME, study, MES_TEST_1)

        # Guardar resultados de test
        guardar_resultados_test(resultados_test, MES_TEST_1)
  
        # Resumen de evaluación en test
        logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
        logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")
    
        logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST 2===")
        
        # Evaluar en test
        resultados_test = evaluar_en_test(conn, SQL_TABLE_NAME, study, MES_TEST_2)

        # Guardar resultados de test
        guardar_resultados_test(resultados_test, MES_TEST_2)
  
        # Resumen de evaluación en test
        logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
        logger.info(f"Ganancia en test: {resultados_test['ganancia_test']:,.0f}")

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()