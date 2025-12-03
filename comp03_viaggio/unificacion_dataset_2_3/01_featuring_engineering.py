import logging
from datetime import datetime
import os
from src.features import *
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
logger.info("Iniciando programa de carga de datos de septiembre")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH_ANT: {DATA_PATH_ANT}")
logger.info(f"DATA_PATH_FE: {DATA_PATH_FE}")
logger.info(f"OUTPUT_PATH_FE: {OUTPUT_PATH_FE}")
logger.info(f"BUCKET_NAME: {BUCKET_NAME}")

### Main ###
def main():
    try:
        conn = duckdb.connect(database=':memory:')
        
        # Configurar acceso a GCS
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
        logger.info("Secret de GCS configurado exitosamente")

        # 1. Cargar datos y crear tabla sql
        conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_ANT, SQL_TABLE_NAME)

        conn = create_new_month_data(DATA_PATH_FE, conn, SQL_TABLE_NAME)

        logger.info("=== GUARDANDO RESULTADO ===")
        save_sql_table_to_parquet(conn, SQL_TABLE_NAME, OUTPUT_PATH_FE)

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()