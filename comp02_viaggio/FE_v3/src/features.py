import duckdb
import logging

logger = logging.getLogger(__name__)


'''def create_sql_table(path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    logger.info(f"Cargando dataset desde {path}")
    conn = duckdb.connect(database=':memory:')
    try:        
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{path}')
        """)
        return conn
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise'''

'''def create_sql_table(path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    
    #Carga un CSV desde 'path' en una tabla DuckDB en memoria y retorna 
    #el objeto de conexión para interactuar con esa tabla.
    
    logger.info(f"Cargando dataset desde {path}")
    conn = duckdb.connect(database=':memory:')
    try:        
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{path}', auto_type_candidates=['VARCHAR', 'FLOAT', 'INTEGER'])
        """)
        return conn
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise'''

def create_sql_table(path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    '''
    Carga un CSV desde 'path' en una tabla DuckDB en memoria y retorna 
    el objeto de conexión para interactuar con esa tabla.
    '''
    logger.info(f"Cargando dataset desde {path}")
    conn = duckdb.connect(database=':memory:')
    try:        
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{path}', auto_type_candidates=['VARCHAR', 'FLOAT', 'INTEGER'])
        """)
        return conn
    
    except Exception as e:
        logger.error(f"Error al cargar el dataset: {e}")
        conn.close()
        raise

def create_sql_table_from_parquet(path: str, table_name: str) -> duckdb.DuckDBPyConnection:
    '''
    Carga un CSV o Parquet desde 'path' en una tabla DuckDB en memoria y retorna 
    el objeto de conexión para interactuar con esa tabla.
    '''
    logger.info(f"Cargando dataset desde {path}")
    conn = duckdb.connect(database=':memory:')
    
    try:
        # Detectar el tipo de archivo por extensión
        if path.lower().endswith('.parquet'):
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM read_parquet('{path}')
            """)
        elif path.lower().endswith('.csv'):
            conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM read_csv_auto('{path}', all_varchar=FALSE)
            """)
        else:
            raise ValueError(f"Formato de archivo no soportado: {path}")
        
        return conn
    
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
    
def create_max_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores máximos por ventana temporal para los atributos especificados y reemplaza la tabla.
    Optimizado usando tabla temporal para mejor performance.
  
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

    logger.info(f"Realizando feature engineering con valores máximos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente")

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
    
    # Generar expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"max({attr}) OVER w AS {attr}_max_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Paso 1: Crear tabla temporal solo con keys y las nuevas features
    logger.info("Creando tabla temporal con features de máximos...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_max AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql_temp)
    
    # Paso 2: Join con la tabla original
    logger.info("Realizando join con tabla original...")
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_max.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_max 
        ON t.numero_de_cliente = temp_max.numero_de_cliente 
        AND t.foto_mes = temp_max.foto_mes
    """
    
    conn.execute(sql_join)
    
    # Limpiar tabla temporal
    conn.execute("DROP TABLE IF EXISTS temp_max")
    logger.info("Features de máximos agregadas exitosamente.")

    return conn


def create_min_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores mínimos por ventana temporal para los atributos especificados y reemplaza la tabla.
    Optimizado usando tabla temporal para mejor performance.
  
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

    logger.info(f"Realizando feature engineering con valores mínimos con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente")

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
    
    # Generar expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"min({attr}) OVER w AS {attr}_min_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Paso 1: Crear tabla temporal solo con keys y las nuevas features
    logger.info("Creando tabla temporal con features de mínimos...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_min AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql_temp)
    
    # Paso 2: Join con la tabla original
    logger.info("Realizando join con tabla original...")
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_min.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_min 
        ON t.numero_de_cliente = temp_min.numero_de_cliente 
        AND t.foto_mes = temp_min.foto_mes
    """
    
    conn.execute(sql_join)
    
    # Limpiar tabla temporal
    conn.execute("DROP TABLE IF EXISTS temp_min")
    logger.info("Features de mínimos agregadas exitosamente.")

    return conn


def create_avg_attributes(conn: duckdb.DuckDBPyConnection, table_name: str, excluir_columnas: list[str], month_window: int = 1) -> duckdb.DuckDBPyConnection:
    """
    Genera variables de valores promedios por ventana temporal para los atributos especificados y reemplaza la tabla.
    Optimizado usando tabla temporal para mejor performance.
  
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

    logger.info(f"Realizando feature engineering con valores promedios con {month_window} meses de ventana temporal para todos los atributos con excepción de las variables con tipo de dato INTEGER o VARCHAR y los {len(excluir_columnas)} atributos excluídos explícitamente")

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

    logger.info(f"Se generarán promedios con ventana temporal para {len(cols_numericas)} columnas.")

    if not cols_numericas:
        logger.warning("No se encontraron columnas numéricas válidas para generar promedios. Devolviendo la conexión sin cambios.")
        return conn
    
    # Generar expresiones SQL para las nuevas columnas
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"avg({attr}) OVER w AS {attr}_avg_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Paso 1: Crear tabla temporal solo con keys y las nuevas features
    logger.info("Creando tabla temporal con features de promedios...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_avg AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql_temp)
    
    # Paso 2: Join con la tabla original
    logger.info("Realizando join con tabla original...")
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_avg.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_avg 
        ON t.numero_de_cliente = temp_avg.numero_de_cliente 
        AND t.foto_mes = temp_avg.foto_mes
    """
    
    conn.execute(sql_join)
    
    # Limpiar tabla temporal
    conn.execute("DROP TABLE IF EXISTS temp_avg")
    logger.info("Features de promedios agregadas exitosamente.")

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

import duckdb
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 1. AGREGACIONES ENTRE FEATURES ORIGINALES
# ============================================================================

def create_sum_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns_to_sum: list[tuple], output_names: list[str]) -> duckdb.DuckDBPyConnection:
    logger.info(f"Creando {len(output_names)} features de suma")
    
    new_cols_sql = []
    for cols, output_name in zip(columns_to_sum, output_names):
        # Crear expresión COALESCE con CAST para manejar VARCHAR
        coalesce_exprs = [f"COALESCE(CAST({col} AS INTEGER), 0)" for col in cols]
        sum_expr = " + ".join(coalesce_exprs)
        new_cols_sql.append(f"{sum_expr} AS {output_name}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Features de suma creadas: {output_names}")
    return conn


def create_diff_features(conn: duckdb.DuckDBPyConnection, table_name: str, column_pairs: list[tuple], output_names: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Resta entre pares de columnas (col1 - col2).
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    column_pairs : list[tuple]
        Lista de tuplas (col1, col2) para calcular col1 - col2
    output_names : list[str]
        Nombres de las columnas resultantes
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando {len(output_names)} features de diferencia")
    
    new_cols_sql = []
    for (col1, col2), output_name in zip(column_pairs, output_names):
        new_cols_sql.append(f"COALESCE({col1}, 0) - COALESCE({col2}, 0) AS {output_name}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Features de diferencia creadas: {output_names}")
    return conn


def create_flag_features(conn: duckdb.DuckDBPyConnection, table_name: str, conditions: list[str], output_names: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Crea flags binarios basados en condiciones.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    conditions : list[str]
        Lista de condiciones SQL
        Ejemplo: ['col1 > 0', 'col2 > 0 AND col3 > 0']
    output_names : list[str]
        Nombres de las columnas resultantes
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando {len(output_names)} features de flags")
    
    new_cols_sql = []
    for condition, output_name in zip(conditions, output_names):
        new_cols_sql.append(f"CASE WHEN {condition} THEN 1 ELSE 0 END AS {output_name}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Features de flags creadas: {output_names}")
    return conn


def create_ratio_features(conn: duckdb.DuckDBPyConnection, table_name: str, numerator_cols: list[str], denominator_cols: list[str], output_names: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Crea ratios entre pares de columnas (numerador / denominador).
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    numerator_cols : list[str]
        Lista de columnas numeradoras
    denominator_cols : list[str]
        Lista de columnas denominadoras
    output_names : list[str]
        Nombres de las columnas resultantes
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando {len(output_names)} features de ratios")
    
    new_cols_sql = []
    for num, den, output_name in zip(numerator_cols, denominator_cols, output_names):
        new_cols_sql.append(f"{num} / NULLIF({den}, 0) AS {output_name}")
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Features de ratios creadas: {output_names}")
    return conn


# ============================================================================
# 2. TENDENCIAS (SLOPE)
# ============================================================================

def create_trend_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str], window: int = 3) -> duckdb.DuckDBPyConnection:
    """
    Calcula tendencia (slope) como (valor_actual - lag_window) / window
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas para calcular tendencia
    window : int
        Ventana temporal (default 3 meses, usa lag_2)
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando tendencias para {len(columns)} columnas con ventana {window}")
    
    new_cols_sql = []
    for col in columns:
        # Para window=3: (valor - lag_2) / 2.0
        lag_n = window - 1
        new_cols_sql.append(f"""
            ({col} - LAG({col}, {lag_n}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) / {lag_n}.0 
            AS {col}_trend_{window}
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Tendencias creadas exitosamente")
    return conn


# ============================================================================
# 3. ACELERACIÓN (SEGUNDA DERIVADA)
# ============================================================================

def create_acceleration_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Calcula aceleración como delta_1 - delta_2
    Requiere que ya existan columnas {col}_delta_1 y {col}_delta_2
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas base (sin el sufijo _delta)
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando aceleraciones para {len(columns)} columnas")
    
    new_cols_sql = []
    for col in columns:
        new_cols_sql.append(f"""
            {col}_delta_1 - {col}_delta_2 AS {col}_accel
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Aceleraciones creadas exitosamente")
    return conn


# ============================================================================
# 4. VOLATILIDAD (COEFICIENTE DE VARIACIÓN)
# ============================================================================

def create_volatility_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str], window: int = 3) -> duckdb.DuckDBPyConnection:
    """
    Calcula volatilidad como stddev / avg sobre ventana temporal
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas para calcular volatilidad
    window : int
        Ventana temporal en meses
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando volatilidad para {len(columns)} columnas con ventana {window}")
    
    new_cols_sql = []
    for col in columns:
        new_cols_sql.append(f"""
            STDDEV({col}) OVER w / NULLIF(ABS(AVG({col}) OVER w), 0) AS {col}_volatility_{window}
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
        )
    """
    
    conn.execute(sql)
    logger.info(f"Volatilidad creada exitosamente")
    return conn


# ============================================================================
# 5. MOMENTUM (RECIENTE VS HISTÓRICO)
# ============================================================================

def create_momentum_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str], recent_window: int = 2, past_start: int = 2, past_end: int = 5) -> duckdb.DuckDBPyConnection:
    """
    Calcula momentum como avg_reciente / avg_pasado
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas para calcular momentum
    recent_window : int
        Ventana reciente (default: últimos 2 meses incluyendo actual)
    past_start : int
        Inicio ventana pasada (default: mes -2)
    past_end : int
        Fin ventana pasada (default: mes -5)
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando momentum para {len(columns)} columnas")
    
    new_cols_sql = []
    for col in columns:
        new_cols_sql.append(f"""
            AVG({col}) OVER w_recent / NULLIF(AVG({col}) OVER w_past, 0) AS {col}_momentum
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
        WINDOW 
            w_recent AS (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes
                ROWS BETWEEN {recent_window - 1} PRECEDING AND CURRENT ROW
            ),
            w_past AS (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes
                ROWS BETWEEN {past_end} PRECEDING AND {past_start} PRECEDING
            )
    """
    
    conn.execute(sql)
    logger.info(f"Momentum creado exitosamente")
    return conn


# ============================================================================
# 6. RACHAS (STREAKS)
# ============================================================================

def create_streak_features(conn: duckdb.DuckDBPyConnection, table_name: str, conditions: list[str], output_names: list[str], window: int = 3) -> duckdb.DuckDBPyConnection:
    """
    Cuenta meses consecutivos donde se cumple una condición
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    conditions : list[str]
        Lista de condiciones SQL
        Ejemplo: ['mrentabilidad <= 0', 'cproductos < LAG(cproductos, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)']
    output_names : list[str]
        Nombres de las columnas resultantes
    window : int
        Ventana temporal para contar
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando {len(output_names)} features de rachas con ventana {window}")
    
    new_cols_sql = []
    for condition, output_name in zip(conditions, output_names):
        new_cols_sql.append(f"""
            SUM(CASE WHEN {condition} THEN 1 ELSE 0 END) 
                OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) 
            AS {output_name}
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Rachas creadas exitosamente")
    return conn


# ============================================================================
# 7. TIME SINCE EVENTS
# ============================================================================

def create_time_since_features(conn: duckdb.DuckDBPyConnection, table_name: str, conditions: list[str], output_names: list[str]) -> duckdb.DuckDBPyConnection:
    """
    Calcula meses desde la última vez que ocurrió un evento
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    conditions : list[str]
        Lista de condiciones que definen el evento
        Ejemplo: ['mconsumototal_tc > 0', 'cpayroll_trx > 0']
    output_names : list[str]
        Nombres de las columnas resultantes
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando {len(output_names)} features de time-since")
    
    new_cols_sql = []
    for condition, output_name in zip(conditions, output_names):
        new_cols_sql.append(f"""
            foto_mes - MAX(CASE WHEN {condition} THEN foto_mes ELSE NULL END) 
                OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS {output_name}
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * {new_cols_str}
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info(f"Time-since features creadas exitosamente")
    return conn


# ============================================================================
# 8. CAMBIOS BRUSCOS (Z-SCORE)
# ============================================================================

def create_sudden_change_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str], threshold: float = 2.0, window: int = 6) -> duckdb.DuckDBPyConnection:
    """
    Detecta cambios bruscos (|z-score| > threshold)
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas para detectar cambios bruscos
    threshold : float
        Umbral de desviaciones estándar (default 2.0)
    window : int
        Ventana histórica para calcular media y std
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando cambios bruscos para {len(columns)} columnas con threshold {threshold}")
    
    # Primero crear z-scores
    zscore_cols_sql = []
    for col in columns:
        zscore_cols_sql.append(f"""
            ({col} - AVG({col}) OVER w_hist) / NULLIF(STDDEV({col}) OVER w_hist, 0) AS zscore_{col}
        """)
    
    zscore_cols_str = ", " + ", ".join(zscore_cols_sql)
    
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_zscores AS
        SELECT 
            *,
            {zscore_cols_str}
        FROM {table_name}
        WINDOW w_hist AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
        )
    """
    
    conn.execute(sql_temp)
    
    # Luego crear flags de sudden changes
    sudden_cols_sql = []
    for col in columns:
        sudden_cols_sql.append(f"""
            CASE WHEN zscore_{col} < -{threshold} THEN 1 ELSE 0 END AS sudden_drop_{col}
        """)
        sudden_cols_sql.append(f"""
            CASE WHEN zscore_{col} > {threshold} THEN 1 ELSE 0 END AS sudden_increase_{col}
        """)
    
    sudden_cols_str = ", " + ", ".join(sudden_cols_sql)
    
    sql_final = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT 
            * EXCLUDE (zscore_{', zscore_'.join(columns)}),
            {sudden_cols_str}
        FROM temp_zscores
    """
    
    conn.execute(sql_final)
    conn.execute("DROP TABLE IF EXISTS temp_zscores")
    
    logger.info(f"Cambios bruscos detectados exitosamente")
    return conn


# ============================================================================
# 9. COMPARACIÓN VS MÁXIMO HISTÓRICO
# ============================================================================

def create_vs_max_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str], window: int = None) -> duckdb.DuckDBPyConnection:
    """
    Calcula ratio valor_actual / max_historico
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas para comparar vs máximo
    window : int or None
        Si None, usa máximo histórico completo
        Si int, usa máximo de últimos N meses
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    window_str = f"_{window}m" if window else "_historico"
    logger.info(f"Creando comparación vs máximo{window_str} para {len(columns)} columnas")
    
    new_cols_sql = []
    for col in columns:
        if window:
            new_cols_sql.append(f"""
                {col} / NULLIF(MAX({col}) OVER w, 0) AS ratio_{col}_vs_max{window_str}
            """)
        else:
            new_cols_sql.append(f"""
                {col} / NULLIF(MAX({col}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 0) 
                AS ratio_{col}_vs_max{window_str}
            """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    if window:
        sql = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * {new_cols_str}
            FROM {table_name}
            WINDOW w AS (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes
                ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
            )
        """
    else:
        sql = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * {new_cols_str}
            FROM {table_name}
        """
    
    conn.execute(sql)
    logger.info(f"Comparación vs máximo creada exitosamente")
    return conn

def create_all_window_attributes(conn, table_name, excluir_columnas, month_window=3):
    """
    Crea MAX, MIN, AVG en una sola pasada
    """
    logger.info(f"Generando todas las window features con ventana {month_window}")
    
    sql_get_cols = f"""
        SELECT name 
        FROM pragma_table_info('{table_name}')
        WHERE type NOT IN ('VARCHAR')
    """
    
    cols_numericas_list = conn.execute(sql_get_cols).fetchall()
    cols_numericas = [c[0] for c in cols_numericas_list if c[0] not in excluir_columnas]
    
    new_cols_sql = []
    for attr in cols_numericas:
        new_cols_sql.append(f"MAX({attr}) OVER w AS {attr}_max_{month_window}")
        new_cols_sql.append(f"MIN({attr}) OVER w AS {attr}_min_{month_window}")
        new_cols_sql.append(f"AVG({attr}) OVER w AS {attr}_avg_{month_window}")
    
    new_cols_str = ", ".join(new_cols_sql)
    
    # Crear tabla temporal
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_window AS
        SELECT 
            numero_de_cliente,
            foto_mes,
            {new_cols_str}
        FROM {table_name} 
        WINDOW w AS (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes
            ROWS BETWEEN {month_window} PRECEDING AND 1 PRECEDING
        )
    """
    
    conn.execute(sql_temp)
    
    # Join con la tabla original
    sql_join = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT t.*, temp_window.* EXCLUDE (numero_de_cliente, foto_mes)
        FROM {table_name} t
        JOIN temp_window 
        ON t.numero_de_cliente = temp_window.numero_de_cliente 
        AND t.foto_mes = temp_window.foto_mes
    """
    
    conn.execute(sql_join)
    conn.execute("DROP TABLE IF EXISTS temp_window")
    
    logger.info(f"Todas las window features creadas exitosamente")
    return conn

def create_behavioral_flags(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Pre-calcula flags comportamentales que se usarán en streaks y time_since.
    Esto mejora significativamente el performance al evitar calcular LAG/MAX múltiples veces.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info("Pre-calculando flags comportamentales para streaks y time_since...")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT 
            *,
            
            -- ====== FLAGS PARA STREAKS ======
            
            -- Flag: rentabilidad negativa
            CASE WHEN mrentabilidad <= 0 THEN 1 ELSE 0 END AS flag_rentabilidad_negativa,
            
            -- Flag: saldo decreciente (comparado con mes anterior)
            CASE WHEN mcuentas_saldo < LAG(mcuentas_saldo, 1) OVER w 
                 THEN 1 ELSE 0 END AS flag_saldo_decreciente,
            
            -- Flag: sin consumo en TC
            CASE WHEN COALESCE(mconsumototal_tc, 0) = 0 THEN 1 ELSE 0 END AS flag_sin_consumo_tc,
            
            -- Flag: sin transacciones digitales
            CASE WHEN COALESCE(transacciones_digitales_total, 0) = 0 
                 THEN 1 ELSE 0 END AS flag_sin_transacciones_digital,
            
            -- Flag: mes inactivo (sin transacciones totales)
            CASE WHEN COALESCE(transacciones_totales, 0) = 0 THEN 1 ELSE 0 END AS flag_inactivo,
            
            -- Flag: perdiendo productos
            CASE WHEN cproductos < LAG(cproductos, 1) OVER w 
                 THEN 1 ELSE 0 END AS flag_perdiendo_productos,
            
            -- Flag: sin payroll
            CASE WHEN COALESCE(payroll_trx_total, 0) = 0 THEN 1 ELSE 0 END AS flag_sin_payroll,
            
            -- Flag: TC en cierre (ya existe, pero lo referencio para claridad)
            tc_en_cierre AS flag_tc_en_cierre,
                   
            -- Flag: endeudamiento creciente
            CASE WHEN endeudamiento_total > LAG(endeudamiento_total, 1) OVER w 
                 THEN 1 ELSE 0 END AS flag_endeudamiento_creciente,
            
            -- Flag: desinvirtiendo
            CASE WHEN inversiones_monto_total < LAG(inversiones_monto_total, 1) OVER w 
                 THEN 1 ELSE 0 END AS flag_desinvirtiendo,
            
            
            -- ====== FLAGS PARA TIME_SINCE ======
            
            -- Flag: tuvo consumo en TC este mes
            CASE WHEN COALESCE(mconsumototal_tc, 0) > 0 THEN 1 ELSE 0 END AS flag_consumo_tc,
            
            -- Flag: tuvo transacciones digitales este mes
            CASE WHEN COALESCE(transacciones_digitales_total, 0) > 0 
                 THEN 1 ELSE 0 END AS flag_trx_digital,
            
            -- Flag: tuvo payroll este mes
            CASE WHEN COALESCE(payroll_trx_total, 0) > 0 THEN 1 ELSE 0 END AS flag_payroll,
            
            -- Flag: hizo plazo fijo este mes
            CASE WHEN cplazo_fijo > 0 THEN 1 ELSE 0 END AS flag_plazo_fijo,
            
            -- Flag: hizo inversión este mes
            CASE WHEN COALESCE(inversiones_count_total, 0) > 0 THEN 1 ELSE 0 END AS flag_inversion,
            
            -- Flag: canceló producto este mes (ya calculado como flag_perdiendo_productos)
            
            -- Flag: es pico de saldo (saldo actual = máximo histórico)
            CASE WHEN mcuentas_saldo = MAX(mcuentas_saldo) OVER w_hist
                 THEN 1 ELSE 0 END AS flag_pico_saldo,
            
            -- Flag: es pico de productos
            CASE WHEN cproductos = MAX(cproductos) OVER w_hist
                 THEN 1 ELSE 0 END AS flag_pico_productos,
            
            -- Flag: es pico de rentabilidad
            CASE WHEN mrentabilidad = MAX(mrentabilidad) OVER w_hist
                 THEN 1 ELSE 0 END AS flag_pico_rentabilidad,
            
            -- Flag: cambió status de TC este mes (de no-cierre a cierre)
            CASE WHEN tc_en_cierre = 1 AND LAG(tc_en_cierre, 1) OVER w = 0
                 THEN 1 ELSE 0 END AS flag_cambio_status_tc
            
        FROM {table_name}
        
        WINDOW 
            w AS (
                PARTITION BY numero_de_cliente 
                ORDER BY foto_mes
            ),
            w_hist AS (
                PARTITION BY numero_de_cliente 
                ORDER BY foto_mes 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
    """
    
    conn.execute(sql)
    logger.info("Flags comportamentales pre-calculados exitosamente")
    
    return conn


def create_active_quarter_feature(conn: duckdb.DuckDBPyConnection, table_name: str) -> duckdb.DuckDBPyConnection:
    """
    Crea la variable active_quarter: cantidad de meses activos en los últimos 3 meses.
    Un mes es "activo" si tiene transacciones_totales > 0.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info("Creando variable active_quarter...")
    
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT 
            *,
            -- Contar meses activos en los últimos 3 meses (incluyendo actual)
            SUM(CASE WHEN flag_inactivo = 0 THEN 1 ELSE 0 END) 
                OVER (
                    PARTITION BY numero_de_cliente 
                    ORDER BY foto_mes 
                    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
                ) AS active_quarter
        FROM {table_name}
    """
    
    conn.execute(sql)
    logger.info("Variable active_quarter creada exitosamente")
    
    return conn

def create_trend_features(conn: duckdb.DuckDBPyConnection, table_name: str, columns: list[str], window: int = 3) -> duckdb.DuckDBPyConnection:
    """
    Calcula tendencia (slope) usando regresión lineal sobre ventana temporal.
    Usa REGR_SLOPE para calcular la pendiente real, no solo cambio promedio.
    
    Parameters:
    -----------
    conn : duckdb.DuckDBPyConnection
        Conexión a DuckDB
    table_name : str
        Nombre de la tabla
    columns : list[str]
        Lista de columnas para calcular tendencia
    window : int
        Ventana temporal (default 3 meses)
    
    Returns:
    --------
    duckdb.DuckDBPyConnection
    """
    
    logger.info(f"Creando tendencias (slope) para {len(columns)} columnas con ventana {window}")
    
    # Paso 1: Crear tabla temporal con row_number para cada cliente
    logger.info("Creando índices temporales para cálculo de slope...")
    sql_temp = f"""
        CREATE TEMP TABLE IF NOT EXISTS temp_with_idx AS
        SELECT 
            *,
            ROW_NUMBER() OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS idx_temporal
        FROM {table_name}
    """
    conn.execute(sql_temp)
    
    # Paso 2: Calcular slopes usando REGR_SLOPE
    logger.info("Calculando slopes usando regresión lineal...")
    new_cols_sql = []
    for col in columns:
        new_cols_sql.append(f"""
            REGR_SLOPE({col}, idx_temporal) 
                OVER (
                    PARTITION BY numero_de_cliente 
                    ORDER BY foto_mes 
                    ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING
                ) AS {col}_trend_{window}
        """)
    
    new_cols_str = ", " + ", ".join(new_cols_sql)
    
    # Paso 3: Crear nueva tabla con slopes
    sql_final = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT * EXCLUDE (idx_temporal) {new_cols_str}
        FROM temp_with_idx
    """
    
    conn.execute(sql_final)
    conn.execute("DROP TABLE IF EXISTS temp_with_idx")
    
    logger.info(f"Tendencias (slope) creadas exitosamente")
    return conn