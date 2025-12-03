import duckdb
import logging
import gc
import os

logger = logging.getLogger(__name__)

def setup_duckdb_connection():
    """Configura conexión DuckDB en memoria con temp en /dev/shm"""
    
    # Crear directorio temporal en RAM (ahora tienes 48GB disponibles)
    temp_dir = '/dev/shm/duckdb_temp'
    os.makedirs(temp_dir, exist_ok=True)
    
    conn = duckdb.connect(database=':memory:')
    
    # NUEVA CONFIGURACIÓN: Aprovechar los 94GB de RAM
    conn.execute(f"SET temp_directory='{temp_dir}'")
    conn.execute("SET memory_limit='60GB'")  # Aumentado de 14GB a 60GB
    conn.execute("SET max_memory='60GB'")
    conn.execute("SET max_temp_directory_size='45GB'")  # Usar hasta 45GB de /dev/shm
    conn.execute("SET threads=12")  # Aumentado de 3 a 16 (ajusta según tus cores)
    conn.execute("SET preserve_insertion_order=false")
    
    logger.info(f"DuckDB configurado:")
    logger.info(f"  - database: :memory:")
    logger.info(f"  - temp_directory: {temp_dir}")
    logger.info(f"  - memory_limit: 60GB")
    logger.info(f"  - max_temp_directory_size: 45GB")
    logger.info(f"  - threads: 12")
    
    return conn, temp_dir

def cleanup_temp_dir(temp_dir):
    """Limpia directorio temporal en RAM"""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal limpiado: {temp_dir}")
    except Exception as e:
        logger.warning(f"Error limpiando directorio temporal: {e}")

def create_sql_table_from_parquet_csv(conn: duckdb.DuckDBPyConnection, path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    '''
    Carga un CSV o Parquet desde 'path' en una tabla DuckDB en memoria y retorna 
    el objeto de conexión para interactuar con esa tabla.
    '''
    logger.info(f"Cargando dataset desde {path}")
   
    try:
        # Detectar el tipo de archivo por extensión
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
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise

def classify_data_types(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Crea una tabla con el esquema clasificado en:
    - 'int_categorico': columnas VARCHAR (categorías codificadas como strings)
    - 'int_numerico': columnas INTEGER/BIGINT
    - 'float': columnas FLOAT/DOUBLE
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla a analizar
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
        Conexión con la tabla de esquema creada
    """
    logger.info(f"Clasificando tipos de datos para tabla {table_name}")
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE schema_clasificado AS
        SELECT 
            name AS variable,
            type AS tipo_original,
            CASE 
                WHEN type IN ('VARCHAR') THEN 'int_categorico'
                WHEN type IN ('INTEGER') THEN 'int_numerico'
                WHEN type IN ('FLOAT'') THEN 'float'
                ELSE 'otro'
            END AS tipo_de_dato
        FROM 
            pragma_table_info('{table_name}')
        ORDER BY 
            name
    """)
    return conn

def get_low_cardinality_columns(conn: duckdb.DuckDBPyConnection, table_name: str, max_unique: int = 10) -> list[str]:
    """
    Retorna lista de columnas que tienen menos de max_unique valores únicos.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla a analizar
    max_unique : int, default=10
        Máximo número de valores únicos
    
    Returns:
    --------
    list[str]
        Lista de nombres de columnas con baja cardinalidad
    """
    logger.info(f"Buscando columnas con menos de {max_unique} valores únicos en {table_name}")
    
    # Obtener todas las columnas
    columnas = conn.execute(f"""
        SELECT name 
        FROM pragma_table_info('{table_name}')
    """).fetchall()
    
    # Construir query que lee la tabla UNA SOLA VEZ
    count_exprs = [f"COUNT(DISTINCT {col[0]}) AS cnt_{i}" 
                   for i, col in enumerate(columnas)]
    
    query = f"""
        SELECT {', '.join(count_exprs)}
        FROM {table_name}
    """
    
    result = conn.execute(query).fetchone()
    
    # Filtrar columnas con baja cardinalidad
    low_cardinality_cols = [
        columnas[i][0] 
        for i, count in enumerate(result) 
        if count < max_unique
    ]
    
    logger.info(f"Encontradas {len(low_cardinality_cols)} columnas con menos de {max_unique} valores únicos")
    
    return low_cardinality_cols

def clase_ternaria(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Genera la tabla con clase_ternaria identificando bajas de clientes.
    Reemplaza la tabla original agregando la columna clase_ternaria.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla a procesar
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
        Conexión con la tabla actualizada
    """
    logger.info(f"Generando clase_ternaria para tabla {table_name}")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH periodos AS (
            SELECT DISTINCT foto_mes FROM {table_name}
        ), clientes AS (
            SELECT DISTINCT numero_de_cliente FROM {table_name}
        ), todo AS (
            SELECT numero_de_cliente, foto_mes 
            FROM clientes CROSS JOIN periodos
        ), clase_ternaria AS (
            SELECT
                c.*,
                IF(c.numero_de_cliente IS NULL, 0, 1) AS mes_0,
                LEAD(mes_0, 1) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_1,
                LEAD(mes_0, 2) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_2,
                IF(mes_1 = 0, 'baja+1', IF(mes_2 = 0, 'baja+2', 'continua')) AS clase_ternaria
            FROM todo t
            LEFT JOIN {table_name} c USING (numero_de_cliente, foto_mes)
        ) 
        SELECT * EXCLUDE (mes_0, mes_1, mes_2)
        FROM clase_ternaria
        WHERE mes_0 = 1
    """
    
    conn.execute(sql)
    logger.info(f"Clase ternaria generada exitosamente para {table_name}")
    return conn

def target_binario(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Genera la tabla con binaria identificando bajas de clientes.
    Reemplaza la tabla original agregando la columna target_binari0.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla a procesar
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
        Conexión con la tabla actualizada
    """
    logger.info(f"Generando target_binario para tabla {table_name}")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH periodos AS (
            SELECT DISTINCT foto_mes FROM {table_name}
        ), clientes AS (
            SELECT DISTINCT numero_de_cliente FROM {table_name}
        ), todo AS (
            SELECT numero_de_cliente, foto_mes 
            FROM clientes CROSS JOIN periodos
        ), target_binario AS (
            SELECT
                c.*,
                IF(c.numero_de_cliente IS NULL, 0, 1) AS mes_0,
                LEAD(mes_0, 1) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_1,
                LEAD(mes_0, 2) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_2,
                IF(mes_1 = 0, 1, IF(mes_2 = 0, 1, 0)) AS target_binario
            FROM todo t
            LEFT JOIN {table_name} c USING (numero_de_cliente, foto_mes)
        ) 
        SELECT * EXCLUDE (mes_0, mes_1, mes_2)
        FROM target_binario
        WHERE mes_0 = 1
    """

    conn.execute(sql)

    logger.info(f"Target binario generada exitosamente para {table_name}")
    return conn
    
def target_ternario(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Genera la tabla con binaria identificando bajas de clientes.
    Reemplaza la tabla original agregando la columna target_ternario.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla a procesar
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
        Conexión con la tabla actualizada
    """
    logger.info(f"Generando target_ternario para tabla {table_name}")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH periodos AS (
            SELECT DISTINCT foto_mes FROM {table_name}
        ), clientes AS (
            SELECT DISTINCT numero_de_cliente FROM {table_name}
        ), todo AS (
            SELECT numero_de_cliente, foto_mes 
            FROM clientes CROSS JOIN periodos
        ), target_ternario AS (
            SELECT
                c.*,
                IF(c.numero_de_cliente IS NULL, 0, 1) AS mes_0,
                LEAD(mes_0, 1) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_1,
                LEAD(mes_0, 2) OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_2,
                IF(mes_1 = 0, 0, IF(mes_2 = 0, 1, 0)) AS target_ternario
                
            FROM todo t
            LEFT JOIN {table_name} c USING (numero_de_cliente, foto_mes)
        ) 
        SELECT * EXCLUDE (mes_0, mes_1, mes_2)
        FROM target_ternario
        WHERE mes_0 = 1
    """

    conn.execute(sql)

    logger.info(f"Target ternario generada exitosamente para {table_name}")
    return conn

def generar_targets(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Genera target_binario y target_ternario en UNA SOLA pasada.
    """
    logger.info(f"Generando targets para tabla {table_name}")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH periodos AS (
            SELECT DISTINCT foto_mes FROM {table_name}
        ), clientes AS (
            SELECT DISTINCT numero_de_cliente FROM {table_name}
        ), todo AS (
            SELECT numero_de_cliente, foto_mes 
            FROM clientes CROSS JOIN periodos
        ), con_flags AS (
            SELECT
                c.*,
                IF(c.numero_de_cliente IS NULL, 0, 1) AS mes_0,
                LEAD(IF(c.numero_de_cliente IS NULL, 0, 1), 1) 
                    OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_1,
                LEAD(IF(c.numero_de_cliente IS NULL, 0, 1), 2) 
                    OVER (PARTITION BY t.numero_de_cliente ORDER BY foto_mes) AS mes_2
            FROM todo t
            LEFT JOIN {table_name} c USING (numero_de_cliente, foto_mes)
        ) 
        SELECT 
            * EXCLUDE (mes_0, mes_1, mes_2),
            IF(mes_1 = 0, 1, IF(mes_2 = 0, 1, 0)) AS target_binario,
            IF(mes_1 = 0, 0, IF(mes_2 = 0, 1, 0)) AS target_ternario
        FROM con_flags
        WHERE mes_0 = 1
    """

    conn.execute(sql)
    logger.info(f"Targets generados exitosamente")
    return conn

def attributes_to_intergers(conn: duckdb.DuckDBPyConnection, table_name: str, cols_to_alter: list[str])-> duckdb.DuckDBPyConnection:
    for col in cols_to_alter:
        conn.execute(f"""
            ALTER TABLE {table_name}
            ALTER COLUMN {col} SET DATA TYPE INTEGER
        """)
    return conn

def drop_columns(conn: duckdb.DuckDBPyConnection, table_name: str, cols_to_drop: list[str])-> duckdb.DuckDBPyConnection:
    for col in cols_to_drop:
        conn.execute(f"ALTER TABLE {table_name} DROP COLUMN {col}")
    return conn

def create_latest_and_earliest_credit_card_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, cols_pairs: list[str]) -> tuple[duckdb.DuckDBPyConnection, list[str]]:
    # 1. Crear la lista de expresiones SQL para las nuevas columnas
    new_cols_sql = []
    new_cols_names = []
    
    for col1, col2, prefix in cols_pairs:
        # Nombres de las nuevas columnas
        latest_name = f"{prefix}_latest"
        earliest_name = f"{prefix}_earliest"
        
        # Expresiones SQL
        latest_expr = f"CAST(greatest({col1}, {col2}) AS INTEGER) AS {latest_name}"
        earliest_expr = f"CAST(least({col1}, {col2}) AS INTEGER) AS {earliest_name}"
        
        new_cols_sql.append(latest_expr)
        new_cols_sql.append(earliest_expr)
        
        # Guardar nombres (esto es Python puro, no pandas)
        new_cols_names.append(latest_name)
        new_cols_names.append(earliest_name)

    # 2. Unir las expresiones con comas
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # 3. Construir la sentencia SQL completa de forma limpia
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT 
            * {new_cols_str}
        FROM 
            {table_name}
    """
    
    conn.execute(sql) 
    return conn, new_cols_names 

def create_sum_credit_card_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, cols_visa: list, cols_master: list)-> duckdb.DuckDBPyConnection:
    conn.execute("""
        CREATE OR REPLACE MACRO suma_sin_null(a, b) AS (
            ifnull(a, 0) + ifnull(b, 0)
        )
    """)

    new_cols_sql = []
    
    for v_col, m_col in zip(cols_visa, cols_master):
         if '_status' not in v_col:
            sufijo = v_col.replace("Visa_", "").replace("visa_", "").replace("Visa", "").replace("visa", "")
            new_cols_sql.append(f"suma_sin_null({v_col},{m_col}) AS {sufijo}_tc")
    
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    conn.execute(sql)    

    return conn

def create_ratio_m_c_attributes(conn: duckdb.DuckDBPyConnection, table_name: str)-> duckdb.DuckDBPyConnection:
    conn.execute("""
        CREATE OR REPLACE MACRO ratio(a, b) AS (a // (b + 0.01))
    """)

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                 pragma_table_info('{table_name}')
        """
    
    columnas_tuples = conn.execute(sql_get_cols).fetchall()
    columnas = [c[0] for c in columnas_tuples]

    # Separar columnas que empiezan con 'm' y 'c'
    cols_m = [c for c in columnas if c.startswith('m')]
    cols_c = [c for c in columnas if c.startswith('c')]

    # Crear diccionario de sufijo a columna
    sufijo_a_m = {c[1:]: c for c in cols_m}  # clave = sufijo sin la primera letra
    sufijo_a_c = {c[1:]: c for c in cols_c}

    # Generar pares donde el sufijo coincide
    pares = [(sufijo_a_m[s], sufijo_a_c[s]) for s in sufijo_a_m if s in sufijo_a_c]

    new_cols_sql = []
    for a, b in pares:
        new_cols_sql.append(f"ratio({a},{b}) AS ratio_{a}_{b}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    conn.execute(sql)    

    return conn

def create_lag_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], cant_lag: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de lag para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de lag agregadas.
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán {cant_lag} lags para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar lags. Devolviendo la conexión sin cambios.")
        return conn
  
    new_cols_sql = []
    for attr in cols_numericas:
        for i in range(1, cant_lag + 1):
            # Agregamos la expresión a la lista
            new_cols_sql.append(f"lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}")
  
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn

def create_delta_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], cant_delta: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de delta para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar deltas.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de delta agregadas.
    """

    logger.info(f"Realizando feature engineering con {cant_delta} deltas para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán {cant_delta} deltas para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar deltas. Devolviendo la conexión sin cambios.")
        return conn
    
    new_cols_sql = []
    for attr in cols_numericas:
        for i in range(1, cant_delta + 1):
            new_cols_sql.append(f"{attr} - lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_delta_{i}")
  
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn

'''def create_max_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores máximos por ventana temporal para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar máximos.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de máximos agregadas.
    """

    logger.info(f"Realizando feature engineering con valores máximos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán máximos con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar máximos. Devolviendo la conexión sin cambios.")
        return conn
    
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"max({attr}) OVER w AS {attr}_max_{month_window}")
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""
  
    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)

    return conn

def create_min_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores mínimos por ventana temporal para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar mínimos.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de mínimos agregadas.
    """

    logger.info(f"Realizando feature engineering con valores mínimos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán mínimos con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar mínimos. Devolviendo la conexión sin cambios.")
        return conn
  
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"min({attr}) OVER w AS {attr}_min_{month_window}")
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""
  
    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)

    return conn

def create_avg_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores promedios por ventana temporal para los atributos especificados y reemplaza la tabla.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar promedios.
    month_window: int, default=1
        Cantidad de meses de la ventana temporal.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de promedios agregadas.
    """

    logger.info(f"Realizando feature engineering con valores promedios con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de de dato INTERGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explicitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán promedios con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar promedios. Devolviendo la conexión sin cambios.")
        return conn
  
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"avg({attr}) OVER w AS {attr}_avg_{month_window}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql) if new_cols_sql else ""

    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn'''

def save_sql_table_to_csv(conn: duckdb.DuckDBPyConnection, table_name: str, path: str) -> None:
    '''
    Guarda la tabla sql en CSV a 'path'.
    '''
    logger.info(f"Guardando tabla en {path}")
    conn.execute(f"""
    COPY {table_name} TO '{path}' (FORMAT CSV, HEADER TRUE)
    """)
    
def create_max_attributes(conn, table_name, excluir_columnas, month_window=3):
    """Versión optimizada para MAX"""
    import logging
    logger = logging.getLogger(__name__)
    
    columnas_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND data_type IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'HUGEINT')
        AND column_name NOT IN ({','.join([f"'{col}'" for col in excluir_columnas])})
    """
    columnas_numericas = [row[0] for row in conn.execute(columnas_query).fetchall()]
    
    if not columnas_numericas:
        return conn
    
    logger.info(f"Creando atributos MAX para {len(columnas_numericas)} columnas")
    
    for idx, col in enumerate(columnas_numericas, 1):
        logger.info(f"  [{idx}/{len(columnas_numericas)}] MAX: {col}")
        
        new_col = f"{col}_max_{month_window}m"
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {new_col} DOUBLE")
        
        sql_update = f"""
            UPDATE {table_name} t
            SET {new_col} = (
                SELECT MAX({col})
                FROM {table_name} t2
                WHERE t2.numero_de_cliente = t.numero_de_cliente
                AND t2.foto_mes < t.foto_mes
                AND t2.foto_mes >= t.foto_mes - {month_window}
            )
        """
        conn.execute(sql_update)
        
        if idx % 5 == 0:
            conn.execute("CHECKPOINT")
    
    return conn

def create_min_attributes(conn, table_name, excluir_columnas, month_window=3):
    """Versión optimizada para MIN"""
    import logging
    logger = logging.getLogger(__name__)
    
    columnas_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND data_type IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'HUGEINT')
        AND column_name NOT IN ({','.join([f"'{col}'" for col in excluir_columnas])})
    """
    columnas_numericas = [row[0] for row in conn.execute(columnas_query).fetchall()]
    
    if not columnas_numericas:
        return conn
    
    logger.info(f"Creando atributos MIN para {len(columnas_numericas)} columnas")
    
    for idx, col in enumerate(columnas_numericas, 1):
        logger.info(f"  [{idx}/{len(columnas_numericas)}] MIN: {col}")
        
        new_col = f"{col}_min_{month_window}m"
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {new_col} DOUBLE")
        
        sql_update = f"""
            UPDATE {table_name} t
            SET {new_col} = (
                SELECT MIN({col})
                FROM {table_name} t2
                WHERE t2.numero_de_cliente = t.numero_de_cliente
                AND t2.foto_mes < t.foto_mes
                AND t2.foto_mes >= t.foto_mes - {month_window}
            )
        """
        conn.execute(sql_update)
        
        if idx % 5 == 0:
            conn.execute("CHECKPOINT")
    
    return conn


def create_avg_attributes(conn, table_name, excluir_columnas, month_window=3):
    """
    Versión ultra-conservadora que minimiza uso de memoria.
    Procesa cada columna con subconsultas correlacionadas en lugar de window functions.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Obtener columnas numéricas
    columnas_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND data_type IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'HUGEINT')
        AND column_name NOT IN ({','.join([f"'{col}'" for col in excluir_columnas])})
    """
    columnas_numericas = [row[0] for row in conn.execute(columnas_query).fetchall()]
    
    if not columnas_numericas:
        logger.info("No hay columnas numéricas para crear atributos AVG")
        return conn
    
    logger.info(f"Creando atributos AVG para {len(columnas_numericas)} columnas (método conservador)")
    
    for idx, col in enumerate(columnas_numericas, 1):
        logger.info(f"  [{idx}/{len(columnas_numericas)}] Procesando: {col}")
        
        try:
            new_col = f"{col}_avg_{month_window}m"
            
            # Agregar columna vacía
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {new_col} DOUBLE")
            
            # Actualizar usando subconsulta correlacionada (más lento pero usa menos memoria)
            sql_update = f"""
                UPDATE {table_name} t
                SET {new_col} = (
                    SELECT AVG({col})
                    FROM {table_name} t2
                    WHERE t2.numero_de_cliente = t.numero_de_cliente
                    AND t2.foto_mes < t.foto_mes
                    AND t2.foto_mes >= t.foto_mes - {month_window}
                )
            """
            conn.execute(sql_update)
            
            # Checkpoint cada 5 columnas
            if idx % 5 == 0:
                conn.execute("CHECKPOINT")
                logger.info(f"    Checkpoint ejecutado")
                
        except Exception as e:
            logger.error(f"Error procesando columna {col}: {e}")
            raise
    
    logger.info(f"Atributos AVG creados exitosamente")
    return conn

def save_sql_table_to_parquet(conn: duckdb.DuckDBPyConnection, table_name: str, path: str) -> None:
    '''
    Guarda la tabla sql en formato Parquet (mucho más eficiente que CSV).
    '''
    logger.info(f"Guardando tabla en formato Parquet a {path}")
    
    # Asegurar que la extensión sea .parquet
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

def create_decile_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de deciles para los atributos especificados.
    Asigna a cada valor un decil (1-10) basado en su posición relativa en foto_mes.
  
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    excluir_columnas : list
        Lista de atributos a excluir para los cuales generar deciles.
  
    Returns:
    --------
    duckdb.DuckDBPyConnection
       Conexión a la tabla DuckDB con los datos con las variables de deciles agregadas.
    """

    logger.info(f"Realizando feature engineering con deciles para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente según la lista {excluir_columnas}")

    sql_get_cols = f"""
            SELECT 
                name 
            FROM 
                pragma_table_info('{table_name}')
            WHERE 
                type NOT IN ('INTEGER', 'VARCHAR')
        """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]

    logger.info(f"Se generarán deciles para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar deciles. Devolviendo la conexión sin cambios.")
        return conn
    
    new_cols_sql = []
    for attr in cols_numericas:
        # NTILE(10) divide en 10 grupos (deciles)
        new_cols_sql.append(f"NTILE(10) OVER (PARTITION BY foto_mes ORDER BY {attr}) AS {attr}_decil")
    
    new_cols_str = ", " + ", ".join(new_cols_sql)

    # Construir la consulta SQL
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    conn.execute(sql)
    return conn

def create_status_binary_attributes(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Genera variables binarias de status para tarjetas Visa y Mastercard.
    Crea columnas indicadoras para cada estado: abierta (0), pre-cierre (6), 
    post-cierre (7) y cerrada (9).
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con los datos.
    table_name : str
        Nombre de la tabla a procesar
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
        Conexión a la tabla DuckDB con las variables de status agregadas.
    """
    
    logger.info(f"Generando variables binarias de status para tarjetas Visa y Mastercard")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT 
            *,
            CASE WHEN visa_status = 0 THEN 1 ELSE 0 END AS visa_status_abierta,
            CASE WHEN visa_status = 6 THEN 1 ELSE 0 END AS visa_status_pcierre,
            CASE WHEN visa_status = 7 THEN 1 ELSE 0 END AS visa_status_pacierre,
            CASE WHEN visa_status = 9 THEN 1 ELSE 0 END AS visa_status_cerrada,
            CASE WHEN master_status = 0 THEN 1 ELSE 0 END AS master_status_abierta,
            CASE WHEN master_status = 6 THEN 1 ELSE 0 END AS master_status_pcierre,
            CASE WHEN master_status = 7 THEN 1 ELSE 0 END AS master_status_pacierre,
            CASE WHEN master_status = 9 THEN 1 ELSE 0 END AS master_status_cerrada
        FROM 
            {table_name}
    """
    
    conn.execute(sql)
    logger.info("Variables binarias de status generadas exitosamente")
    
    return conn

def create_delta_attributes_chunked(conn, table_name, excluir_columnas, cant_delta=2):
    """
    Versión optimizada que procesa deltas en chunks pequeños
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Obtener columnas numéricas
    columnas_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND data_type IN ('INTEGER', 'BIGINT', 'DOUBLE', 'FLOAT', 'DECIMAL', 'HUGEINT')
        AND column_name NOT IN ({','.join([f"'{col}'" for col in excluir_columnas])})
    """
    columnas_numericas = [row[0] for row in conn.execute(columnas_query).fetchall()]
    
    if not columnas_numericas:
        return conn
    
    logger.info(f"Creando atributos DELTA para {len(columnas_numericas)} columnas")
    
    # Procesar de a 5 columnas para minimizar memoria
    batch_size = 5
    
    for i in range(0, len(columnas_numericas), batch_size):
        batch = columnas_numericas[i:i+batch_size]
        logger.info(f"  Procesando lote {i//batch_size + 1}/{(len(columnas_numericas)-1)//batch_size + 1}: {len(batch)} columnas")
        
        for lag in range(1, cant_delta + 1):
            for col in batch:
                new_col = f"{col}_delta_{lag}"
                lag_col = f"{col}_lag_{lag}"
                
                try:
                    # Agregar columna
                    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {new_col} DOUBLE")
                    
                    # Calcular delta
                    conn.execute(f"""
                        UPDATE {table_name}
                        SET {new_col} = {col} - {lag_col}
                        WHERE {lag_col} IS NOT NULL
                    """)
                    
                except Exception as e:
                    logger.error(f"Error procesando {col}_delta_{lag}: {e}")
                    raise
        
        # Checkpoint cada lote
        conn.execute("CHECKPOINT")
        logger.info(f"    Checkpoint ejecutado")
    
    logger.info(f"Atributos DELTA creados exitosamente")
    return conn