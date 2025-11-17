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
logger.info("Iniciando programa de optimización con log fechado")

### Manejo de Configuración en YAML ###
logger.info("Configuración cargada desde YAML")
logger.info(f"STUDY_NAME: {STUDY_NAME}")
logger.info(f"DATA_PATH_FE: {DATA_PATH_FE}")

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO INGENIERIA DE ATRIBUTOS CON CONFIGURACIÓN YAML ===")

    conn = None # Inicializamos la conexión a None
    try:  
        # 1. Cargar datos y crear tabla sql
        conn = create_sql_table(DATA_PATH_FE, SQL_TABLE_NAME)

        conn = create_status_binary_attributes(conn, SQL_TABLE_NAME)
        cols_to_drop = ["master_status", "visa_status"]
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_to_drop)
    
        # 2. Columnas con baja cardinalidad
        low_cardinality_cols = get_low_cardinality_columns(conn, SQL_TABLE_NAME, max_unique=10)

        # 3. Crear atributos tipo fecha mayor y menor para las tarjetas de crédito
        column_pairs = [
        ("Master_Finiciomora", "Visa_Finiciomora", "tc_finiciomora"),
        ("Master_Fvencimiento", "Visa_Fvencimiento", "tc_fvencimiento"),
        ("Master_fultimo_cierre", "Visa_fultimo_cierre", "tc_fultimocierre"),
        ("Master_fechaalta", "Visa_fechaalta", "tc_fechaalta"),
        ]
        conn, cols_tc_fecha = create_latest_and_earliest_credit_card_attributes(conn, SQL_TABLE_NAME, column_pairs)

        # 4. Borrar columnas tipo fecha individuales de las tarjetas de crédito máster y visa
        cols_to_drop = [
        "Master_Finiciomora", "Visa_Finiciomora",
        "Master_Fvencimiento", "Visa_Fvencimiento",
        "Master_fultimo_cierre", "Visa_fultimo_cierre",
        "Master_fechaalta", "Visa_fechaalta"
        ]
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_to_drop)

        # 5. Crear atributos tipo suma para las tarjetas de crédito
        sql_get_cols_visa = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{SQL_TABLE_NAME}')
            WHERE 
                name ILIKE '%visa%'
        """
        
        cols_visa = conn.execute(sql_get_cols_visa).fetchall()

        sql_get_cols_master = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{SQL_TABLE_NAME}')
            WHERE 
                name ILIKE '%master%'
        """

        cols_master = conn.execute(sql_get_cols_master).fetchall()

        cols_visa_str = [c[0] for c in cols_visa]
        cols_master_str = [c[0] for c in cols_master]

        conn = create_sum_credit_card_attributes(conn, SQL_TABLE_NAME, cols_visa_str, cols_master_str)

        # 6. Borrar atributos individuales usandos para crear los atributos tipo suma para las tarjetas de crédito
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_visa_str+cols_master_str)

        # 7. Crear atributos tipo ratio entre pares de variables m_ y c_
        conn = create_ratio_m_c_attributes(conn, SQL_TABLE_NAME)

        # 8. Crear atributos tipo lag
        excluir_columnas_lag = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_tc_fecha + low_cardinality_cols
        conn = create_lag_attributes(conn, SQL_TABLE_NAME, excluir_columnas_lag, cant_lag = 2)

        # 9. Crear atributos tipo delta
        sql_get_cols_lag = f"""
            SELECT name 
            FROM pragma_table_info('{SQL_TABLE_NAME}')
            WHERE
                name LIKE '%lag_1'
                OR name LIKE '%lag_2'
        """
        
        cols_lag = conn.execute(sql_get_cols_lag).fetchall()
        cols_lag_list = [c[0] for c in cols_lag]
        excluir_columnas_delta = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_list + cols_tc_fecha + low_cardinality_cols
        conn = create_delta_attributes(conn, SQL_TABLE_NAME, excluir_columnas_delta, cant_delta = 2)

        # 10. Crear atributos tipo máximos ventana
        sql_get_cols_lag_delta = f"""
            SELECT name 
            FROM pragma_table_info('{SQL_TABLE_NAME}')
            WHERE
                name LIKE '%lag_1'
                OR name LIKE '%lag_2'
                OR name LIKE '%delta_1'
                OR name LIKE '%delta_2'            
        """
        
        cols_lag_delta = conn.execute(sql_get_cols_lag_delta).fetchall()
        cols_lag_delta_list = [c[0] for c in cols_lag_delta]
        excluir_columnas_max = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_delta_list + cols_tc_fecha + low_cardinality_cols
        conn = create_max_attributes(conn, SQL_TABLE_NAME, excluir_columnas_max, month_window = 3)

        # 11. Crear atributos tipo mínimos ventana
        sql_get_cols_lag_delta_max = f"""
            SELECT name 
            FROM pragma_table_info('{SQL_TABLE_NAME}')
            WHERE
                name LIKE '%lag_1'
                OR name LIKE '%lag_2'
                OR name LIKE '%delta_1'
                OR name LIKE '%delta_2'
                OR name LIKE '%max_3'            
        """
        
        cols_lag_delta_max = conn.execute(sql_get_cols_lag_delta_max).fetchall()
        cols_lag_delta_max_list = [c[0] for c in cols_lag_delta_max]
        excluir_columnas_min = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_delta_max_list + cols_tc_fecha + low_cardinality_cols
        conn = create_min_attributes(conn, SQL_TABLE_NAME, excluir_columnas_min, month_window = 3)

        # 12. Crear atributos tipo promedio ventana
        sql_get_cols_lag_delta_max_min = f"""
            SELECT name 
            FROM pragma_table_info('{SQL_TABLE_NAME}')
            WHERE
                name LIKE '%lag_1'
                OR name LIKE '%lag_2'
                OR name LIKE '%delta_1'
                OR name LIKE '%delta_2'
                OR name LIKE '%max_3'  
                OR name LIKE '%min_3'           
        """
        
        cols_lag_delta_max_min = conn.execute(sql_get_cols_lag_delta_max_min).fetchall()
        cols_lag_delta_max_min_list = [c[0] for c in cols_lag_delta_max_min]
        excluir_columnas_avg = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_lag_delta_max_min_list + cols_tc_fecha + low_cardinality_cols
        conn = create_avg_attributes(conn, SQL_TABLE_NAME, excluir_columnas_avg, month_window = 3)

        # 13. generar targets para optimización y entrenamiento
        conn = generar_targets(conn, SQL_TABLE_NAME)

        # 14. Guardar el CSV con el FE
        save_sql_table_to_parquet(conn,SQL_TABLE_NAME, OUTPUT_PATH_FE)

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")

if __name__ == "__main__":
    main()