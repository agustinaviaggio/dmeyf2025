import duckdb
import logging
import gc

logger = logging.getLogger(__name__)

def create_sql_table_from_parquet_csv(conn: duckdb.DuckDBPyConnection, path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    '''
    Carga un CSV o Parquet desde 'path' en una tabla DuckDB en memoria y retorna 
    el objeto de conexión para interactuar con esa tabla.
    '''
    logger.info(f"Cargando dataset desde {path}")
   
    try:
        if path.lower().endswith('.parquet'):
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM read_parquet('{path}')
            """)
        elif path.lower().endswith('.csv') or path.lower().endswith('.csv.gz'):
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM read_csv_auto('{path}', auto_type_candidates=['VARCHAR', 'FLOAT', 'INTEGER'])
            """)
        else:
            raise ValueError(f"Formato de archivo no soportado: {path}")
        
        gc.collect()
        
        return conn
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise

def create_new_month_data(path: str, conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    '''
    Carga un CSV desde 'path' y agrega sus filas a una tabla DuckDB existente
    (UNION), retornando el objeto de conexión actualizado.
    
    Parameters:
    -----------
    path : str
        Ruta al archivo CSV con las nuevas filas
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla existente a la que se agregarán filas
    '''
    logger.info(f"Agregando filas desde {path} a tabla {table_name}")
    
    try:
        temp_table = f"{table_name}_temp"

        conn.execute(f"""
            CREATE OR REPLACE TABLE {temp_table} AS
            SELECT *
            FROM read_csv_auto('{path}', auto_type_candidates=['VARCHAR', 'FLOAT', 'INTEGER'])
        """)
        
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM {table_name}
            UNION
            SELECT * FROM {temp_table}
        """)
        
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        
        gc.collect()
        
        logger.info(f"Filas agregadas exitosamente a {table_name}")
        return conn
    
    except Exception as e:
        logger.error(f"Error al agregar filas desde {path}: {e}")
        try:
            conn.execute(f"DROP TABLE IF EXISTS {temp_table}")
        except:
            pass
        raise

def save_sql_table_to_parquet(conn: duckdb.DuckDBPyConnection, table_name: str, path: str) -> None:
    '''
    Guarda la tabla sql en formato Parquet.
    '''
    logger.info(f"Guardando tabla en formato Parquet a {path}")

    if not path.endswith('.parquet'):
        if path.endswith('.csv'):
            path = path.replace('.csv', '.parquet')
        else:
            path = path + '.parquet'
    
    conn.execute(f"""
        COPY {table_name} TO '{path}' 
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    
    logger.info(f"Tabla guardada exitosamente en {path}")
