import logging
from datetime import datetime
import os
import duckdb
from src.features import save_sql_table_to_parquet
from src.features_v4 import generar_targets
from src.config import *

### Configuración de logging ###
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_targets_{STUDY_NAME}_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler("logs/" + nombre_log),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Script para generar targets desde dataset con features"""
    logger.info("=== GENERANDO TARGETS ===")
    
    temp_dir = '/dev/shm/duckdb_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    conn = None
    
    try:
        # Configuración MUY conservadora para esta operación pesada
        conn = duckdb.connect(database=':memory:')
        conn.execute(f"SET temp_directory='{temp_dir}'")
        conn.execute("SET memory_limit='70GB'")  # Más memoria para targets
        conn.execute("SET max_memory='70GB'")
        conn.execute("SET max_temp_directory_size='45GB'")
        conn.execute("SET threads=8")  # Menos threads, más memoria por thread
        conn.execute("SET preserve_insertion_order=false")
        
        logger.info("DuckDB configurado para generación de targets")
        
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
        
        # Cargar dataset SIN targets desde GCS
        input_path = OUTPUT_PATH_FE.replace('.parquet', '_sin_targets.parquet')
        logger.info(f"Cargando datos desde: {input_path}")
        
        conn.execute(f"""
            CREATE TABLE {SQL_TABLE_NAME} AS 
            SELECT * FROM read_parquet('{input_path}')
        """)
        
        logger.info("Datos cargados exitosamente")
        
        # Generar targets
        logger.info("Generando targets...")
        conn = generar_targets(conn, SQL_TABLE_NAME)
        
        # Guardar resultado final CON targets
        logger.info("Guardando resultado final CON targets...")
        save_sql_table_to_parquet(conn, SQL_TABLE_NAME, OUTPUT_PATH_FE)
        
        logger.info("=== TARGETS GENERADOS EXITOSAMENTE ===")
        logger.info(f"Dataset final guardado en: {OUTPUT_PATH_FE}")
        
    except Exception as e:
        logger.error(f"Error generando targets: {e}")
        raise
        
    finally:
        if conn:
            conn.close()
            logger.info("Conexión cerrada")
        
        # Limpiar temp
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    main()