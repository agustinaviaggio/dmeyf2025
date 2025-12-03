import logging
from datetime import datetime
import os
import gc
from src.features_v4 import *
from src.features import create_sql_table_from_parquet_csv
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

def cleanup_temp_dir(temp_dir):
    """Limpia directorio temporal"""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Directorio temporal limpiado: {temp_dir}")
    except Exception as e:
        logger.warning(f"Error limpiando temp_dir: {e}")

def save_checkpoint(conn, table_name, checkpoint_path):
    """Guarda checkpoint a disco"""
    logger.info(f"=== GUARDANDO CHECKPOINT: {checkpoint_path} ===")
    try:
        conn.execute(f"""
            COPY {table_name} 
            TO '{checkpoint_path}' 
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """)
        logger.info(f"Checkpoint guardado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error guardando checkpoint: {e}")
        return False

def load_checkpoint(conn, table_name, checkpoint_path):
    """Carga checkpoint desde disco"""
    logger.info(f"=== CARGANDO CHECKPOINT: {checkpoint_path} ===")
    try:
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM read_parquet('{checkpoint_path}')
        """)
        logger.info(f"Checkpoint cargado exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error cargando checkpoint: {e}")
        return False

def checkpoint_exists(conn, checkpoint_path):
    """Verifica si existe el checkpoint usando DuckDB"""
    try:
        result = conn.execute(f"""
            SELECT COUNT(*) 
            FROM read_parquet('{checkpoint_path}')
            LIMIT 1
        """).fetchone()
        logger.info(f"Checkpoint encontrado: {checkpoint_path}")
        return True
    except Exception as e:
        logger.info(f"Checkpoint no existe: {checkpoint_path}")
        return False

### Main ###
def main():
    """Pipeline principal con optimización usando configuración YAML."""
    logger.info("=== INICIANDO INGENIERIA DE ATRIBUTOS CON CONFIGURACIÓN YAML ===")
    
    conn = None
    temp_dir = None
    
    try: 
        temp_dir = '/dev/shm/duckdb_temp'
        os.makedirs(temp_dir, exist_ok=True)
        
        conn = duckdb.connect(database=':memory:')
        
        conn.execute(f"SET temp_directory='{temp_dir}'")
        conn.execute("SET memory_limit='60GB'")
        conn.execute("SET max_memory='60GB'")
        conn.execute("SET max_temp_directory_size='45GB'")
        conn.execute("SET threads=12")
        conn.execute("SET preserve_insertion_order=false")
        
        logger.info(f"DuckDB configurado: temp_dir={temp_dir}, memory=60GB, threads=12")

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

        checkpoint_before_active = OUTPUT_PATH_FE.replace('.parquet', '_checkpoint_before_active_quarter.parquet')
        checkpoint_after_active = OUTPUT_PATH_FE.replace('.parquet', '_checkpoint_after_active_quarter.parquet')
        checkpoint_after_trends = OUTPUT_PATH_FE.replace('.parquet', '_checkpoint_after_trends.parquet')
        checkpoint_after_accel = OUTPUT_PATH_FE.replace('.parquet', '_checkpoint_after_accel.parquet')
        checkpoint_after_momentum = OUTPUT_PATH_FE.replace('.parquet', '_checkpoint_after_momentum.parquet')
        checkpoint_after_streaks = OUTPUT_PATH_FE.replace('.parquet', '_checkpoint_after_streaks.parquet')

        skip_to_active = False
        skip_to_trends = False
        skip_to_accel = False
        skip_to_momentum = False
        skip_to_streaks = False
        skip_to_time_since = False
        checkpoint_loaded = False
        
        if checkpoint_exists(conn, checkpoint_after_streaks):
            logger.info("=== CHECKPOINT DESPUES DE STREAKS ENCONTRADO - CARGANDO ===")
            load_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_streaks)
            skip_to_time_since = True
            checkpoint_loaded = True
            
        elif checkpoint_exists(conn, checkpoint_after_momentum):
            logger.info("=== CHECKPOINT DESPUES DE MOMENTUM ENCONTRADO - CARGANDO ===")
            load_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_momentum)
            skip_to_streaks = True
            checkpoint_loaded = True
            
        elif checkpoint_exists(conn, checkpoint_after_accel):
            logger.info("=== CHECKPOINT DESPUES DE ACCEL ENCONTRADO - CARGANDO ===")
            load_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_accel)
            skip_to_momentum = True
            checkpoint_loaded = True
            
        elif checkpoint_exists(conn, checkpoint_after_trends):
            logger.info("=== CHECKPOINT DESPUES DE TRENDS ENCONTRADO - CARGANDO ===")
            load_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_trends)
            skip_to_accel = True
            checkpoint_loaded = True
            
        elif checkpoint_exists(conn, checkpoint_after_active):
            logger.info("=== CHECKPOINT DESPUES DE active_quarter ENCONTRADO - CARGANDO ===")
            load_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_active)
            skip_to_trends = True
            checkpoint_loaded = True
            
        elif checkpoint_exists(conn, checkpoint_before_active):
            logger.info("=== CHECKPOINT ANTES DE active_quarter ENCONTRADO - CARGANDO ===")
            load_checkpoint(conn, SQL_TABLE_NAME, checkpoint_before_active)
            skip_to_active = True
            checkpoint_loaded = True
            
        else:
            logger.info("=== NO HAY CHECKPOINTS - EJECUTANDO TODO EL PIPELINE ===")
        
        if not checkpoint_loaded:
            logger.info("=== NO HAY CHECKPOINT - EJECUTANDO TODO EL PIPELINE ===")
            
            # Cargar datos y crear tabla sql
            conn = create_sql_table_from_parquet_csv(conn, DATA_PATH_FE, SQL_TABLE_NAME)

            logger.info("Verificando y convirtiendo columnas VARCHAR numéricas...")
            varchar_cols = conn.execute(f"""
                SELECT name 
                FROM pragma_table_info('{SQL_TABLE_NAME}')
                WHERE type = 'VARCHAR'
                AND name NOT IN ('numero_de_cliente', 'clase_ternaria')
            """).fetchall()

            for col_tuple in varchar_cols:
                col = col_tuple[0]
                try:
                    conn.execute(f"""
                        ALTER TABLE {SQL_TABLE_NAME}
                        ALTER COLUMN {col} TYPE DOUBLE USING TRY_CAST({col} AS DOUBLE)
                    """)
                    logger.info(f"Columna {col} convertida de VARCHAR a DOUBLE")
                except Exception as e:
                    logger.debug(f"Columna {col} permanece como VARCHAR: {e}")

            conn = create_status_binary_attributes(conn, SQL_TABLE_NAME)
            cols_to_drop = ["master_status", "visa_status"]
            conn = drop_columns(conn, SQL_TABLE_NAME, cols_to_drop)
        
            # Columnas con baja cardinalidad
            low_cardinality_cols = get_low_cardinality_columns(conn, SQL_TABLE_NAME, max_unique=5)

            # Crear atributos tipo fecha mayor y menor para las tarjetas de crédito
            column_pairs = [
            ("Master_Finiciomora", "Visa_Finiciomora", "tc_finiciomora"),
            ("Master_Fvencimiento", "Visa_Fvencimiento", "tc_fvencimiento"),
            ("Master_fultimo_cierre", "Visa_fultimo_cierre", "tc_fultimocierre"),
            ("Master_fechaalta", "Visa_fechaalta", "tc_fechaalta"),
            ]
            conn, cols_tc_fecha = create_latest_and_earliest_credit_card_attributes(conn, SQL_TABLE_NAME, column_pairs)

            # Borrar columnas tipo fecha individuales de las tarjetas de crédito máster y visa
            cols_to_drop = [
            "Master_Finiciomora", "Visa_Finiciomora",
            "Master_Fvencimiento", "Visa_Fvencimiento",
            "Master_fultimo_cierre", "Visa_fultimo_cierre",
            "Master_fechaalta", "Visa_fechaalta"
            ]
            conn = drop_columns(conn, SQL_TABLE_NAME, cols_to_drop)

            # Crear atributos tipo suma para las tarjetas de crédito
            sql_get_cols_visa = f"""
                SELECT 
                    name 
                FROM 
                    pragma_table_info('{SQL_TABLE_NAME}')
                WHERE 
                    name ILIKE '%visa%'
                AND name NOT ILIKE '%status%'
            """
            
            cols_visa = conn.execute(sql_get_cols_visa).fetchall()

            sql_get_cols_master = f"""
                SELECT 
                    name 
                FROM 
                    pragma_table_info('{SQL_TABLE_NAME}')
                WHERE 
                    name ILIKE '%master%'
                AND name NOT ILIKE '%status%'
            """

            cols_master = conn.execute(sql_get_cols_master).fetchall()

            cols_visa_str = [c[0] for c in cols_visa]
            cols_master_str = [c[0] for c in cols_master]

            conn = create_sum_credit_card_attributes(conn, SQL_TABLE_NAME, cols_visa_str, cols_master_str)

            # Borrar atributos individuales usandos para crear los atributos tipo suma para las tarjetas de crédito
            conn = drop_columns(conn, SQL_TABLE_NAME, cols_visa_str+cols_master_str)

            # Crear atributos tipo ratio entre pares de variables m_ y c_
            conn = create_ratio_m_c_attributes(conn, SQL_TABLE_NAME)

            # Deuda total en préstamos
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('mprestamos_personales', 'mprestamos_prendarios', 'mprestamos_hipotecarios'),
                    ('cprestamos_personales', 'cprestamos_prendarios', 'cprestamos_hipotecarios')
                ],
                output_names=['deuda_total_prestamos', 'prestamos_count_total']
            )

            # Endeudamiento total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('msaldototal_tc', 'deuda_total_prestamos')
                ],
                output_names=['endeudamiento_total']
            )

            # Inversiones total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('mplazo_fijo_dolares', 'mplazo_fijo_pesos'),
                    ('mplazo_fijo_dolares', 'mplazo_fijo_pesos', 'minversion1_pesos', 'minversion1_dolares', 'minversion2'),
                    ('cplazo_fijo', 'cinversion1', 'cinversion2')
                ],
                output_names=['plazo_fijo_total', 'inversiones_monto_total', 'inversiones_count_total']
            )

            # Seguros total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('cseguro_vida', 'cseguro_auto', 'cseguro_vivienda', 'cseguro_accidentes_personales')
                ],
                output_names=['seguros_total']
            )

            # Payroll total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('mpayroll', 'mpayroll2'),
                    ('cpayroll_trx', 'cpayroll2_trx')
                ],
                output_names=['payroll_monto_total', 'payroll_trx_total']
            )

            # Pagos de servicios total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('cpagodeservicios', 'cpagomiscuentas'),
                    ('mpagodeservicios', 'mpagomiscuentas')
                ],
                output_names=['pagos_servicios_count_total', 'pagos_servicios_monto_total']
            )

            # Comisiones total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('ccomisiones_mantenimiento', 'ccomisiones_otras'),
                    ('mcomisiones_mantenimiento', 'mcomisiones_otras')
                ],
                output_names=['comisiones_count_total', 'comisiones_monto_total']
            )

            # Forex balance (diferencia)
            conn = create_diff_features(
                conn, SQL_TABLE_NAME,
                column_pairs=[('mforex_buy', 'mforex_sell')],
                output_names=['forex_balance']
            )

             # Transferencias balance
            conn = create_diff_features(
                conn, SQL_TABLE_NAME,
                column_pairs=[
                    ('ctransferencias_recibidas', 'ctransferencias_emitidas'),
                    ('mtransferencias_recibidas', 'mtransferencias_emitidas')
                ],
                output_names=['transferencias_balance_count', 'transferencias_balance_monto']
            )

            # 13.15 Canales digitales
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('chomebanking_transacciones', 'cmobile_app_trx'),
                    ('thomebanking', 'tmobile_app')
                ],
                output_names=['transacciones_digitales_total', 'canales_digitales_activos']
            )

            # Canales físicos
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('ccajas_transacciones', 'ccajas_consultas', 'ccajas_depositos', 'ccajas_extracciones', 'ccajas_otras')
                ],
                output_names=['transacciones_cajas_total']
            )

            # ATM total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('catm_trx', 'catm_trx_other'),
                    ('matm', 'matm_other')
                ],
                output_names=['atm_trx_total', 'atm_monto_total']
            )

            # Transacciones totales
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('transacciones_digitales_total', 'ccallcenter_transacciones', 'transacciones_cajas_total', 'atm_trx_total')
                ],
                output_names=['transacciones_totales']
            )

            # Cuentas total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('ccuenta_corriente', 'ccaja_ahorro'),
                    ('mcuenta_corriente', 'mcuenta_corriente_adicional', 'mcaja_ahorro', 'mcaja_ahorro_adicional')
                ],
                output_names=['cuentas_total', 'saldo_pesos_total']
            )

            # Margen total
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('mactivos_margen', 'mpasivos_margen')
                ],
                output_names=['margen_total']
            )

            # Activos totales y patrimonio
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('mcuentas_saldo', 'inversiones_monto_total')
                ],
                output_names=['activos_totales']
            )

            conn = create_diff_features(
                conn, SQL_TABLE_NAME,
                column_pairs=[('activos_totales', 'endeudamiento_total')],
                output_names=['patrimonio_neto']
            )

            # Actividad por tipo
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('ctransferencias_emitidas', 'ctransferencias_recibidas', 'cextraccion_autoservicio', 'pagos_servicios_count_total'),
                    ('cplazo_fijo', 'cinversion1', 'cinversion2', 'cforex')
                ],
                output_names=['actividad_transaccional', 'actividad_inversora']
            )

            # FLAGS de productos
            conn = create_flag_features(
                conn, SQL_TABLE_NAME,
                conditions=[
                    'ccuenta_corriente > 0',
                    'ccaja_ahorro > 0',
                    'master_status_abierta = 1 OR visa_status_abierta = 1',
                    'prestamos_count_total > 0',
                    'inversiones_count_total > 0',
                    'seguros_total > 0',
                    'payroll_trx_total > 0',
                    'master_status_pcierre = 1 OR visa_status_pcierre = 1 OR master_status_pacierre = 1 OR visa_status_pacierre = 1',
                    'master_status_cerrada = 1 OR visa_status_cerrada = 1'
                ],
                output_names=[
                    'tiene_cuenta_corriente', 'tiene_caja_ahorro', 'tiene_tc_activa',
                    'tiene_prestamos', 'tiene_inversiones', 'tiene_seguros', 'tiene_payroll',
                    'tc_en_cierre', 'tiene_tc_cerrada'
                ]
            )

            # Diversificación de productos
            conn = create_sum_features(
                conn, SQL_TABLE_NAME,
                columns_to_sum=[
                    ('tiene_cuenta_corriente', 'tiene_caja_ahorro', 'tiene_tc_activa', 'tiene_prestamos', 'tiene_inversiones', 'tiene_seguros')
                ],
                output_names=['diversificacion_productos']
            )

            # Ratios de endeudamiento
            conn = create_ratio_features(
                conn, SQL_TABLE_NAME,
                numerator_cols=['endeudamiento_total', 'msaldototal_tc', 'mconsumototal_tc'],
                denominator_cols=['mcuentas_saldo', 'mlimitecompra_tc', 'mlimitecompra_tc'],
                output_names=['ratio_endeudamiento_vs_saldo', 'ratio_saldo_tc_vs_limite', 'ratio_consumo_tc_vs_limite']
            )

            # Ratios de comportamiento de pago
            conn = create_ratio_features(
                conn, SQL_TABLE_NAME,
                numerator_cols=['mpagado_tc', 'mpagado_tc', 'madelantopesos_tc'],
                denominator_cols=['mconsumototal_tc', 'mpagominimo_tc', 'mconsumototal_tc'],
                output_names=['ratio_pagos_vs_consumo_tc', 'ratio_pagado_vs_minimo_tc', 'ratio_adelantos_vs_consumo_tc']
            )

            # Ratios de inversión
            conn = create_ratio_features(
                conn, SQL_TABLE_NAME,
                numerator_cols=['inversiones_monto_total', 'inversiones_monto_total'],
                denominator_cols=['mcuentas_saldo', 'activos_totales'],
                output_names=['ratio_inversiones_vs_saldo', 'ratio_inversiones_vs_activos']
            )

            # Crear atributos tipo lag
            excluir_columnas_lag = ['numero_de_cliente', 'foto_mes', 'cliente_edad', 'cliente_antiguedad'] + cols_tc_fecha + low_cardinality_cols
            conn = create_lag_attributes(conn, SQL_TABLE_NAME, excluir_columnas_lag, cant_lag = 2)

            # Crear atributos tipo delta
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

            # Crear atributos tipo máximos ventana
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
            
            conn = create_all_window_attributes(conn, SQL_TABLE_NAME, excluir_columnas_max, month_window = 3)
            conn = create_behavioral_flags(conn, SQL_TABLE_NAME)
            
            # GUARDAR CHECKPOINT ANTES DE active_quarter
            logger.info("=== GUARDANDO CHECKPOINT ANTES DE active_quarter ===")
            save_checkpoint(conn, SQL_TABLE_NAME, checkpoint_before_active)
            
            logger.info("=== LIBERANDO MEMORIA ANTES DE active_quarter ===")
            gc.collect()
        
        # ACTIVE_QUARTER
        if not (skip_to_trends or skip_to_accel or skip_to_momentum or skip_to_streaks or skip_to_time_since):
            logger.info("=== CREANDO active_quarter ===")
            conn = create_active_quarter_feature(conn, SQL_TABLE_NAME)
            
            logger.info("=== GUARDANDO CHECKPOINT DESPUES DE active_quarter ===")
            save_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_active)
            gc.collect()
        
        # TRENDS
        if not (skip_to_accel or skip_to_momentum or skip_to_streaks or skip_to_time_since):
            logger.info("=== CREANDO TRENDS ===")
            vars_criticas_trend = [
                'mrentabilidad', 'mcuentas_saldo', 'mconsumototal_tc', 'cproductos',
                'transacciones_digitales_total', 'payroll_monto_total', 'inversiones_monto_total',
                'endeudamiento_total', 'margen_total',
                'cuentas_total', 'seguros_total', 'diversificacion_productos',
                'actividad_transaccional', 'actividad_inversora', 'comisiones_monto_total'
            ]
            ratios_trend = [
                'ratio_endeudamiento_vs_saldo', 'ratio_consumo_tc_vs_limite', 
                'ratio_pagos_vs_consumo_tc', 'ratio_inversiones_vs_saldo'
            ]
            conn = create_trend_features(conn, SQL_TABLE_NAME, vars_criticas_trend + ratios_trend, window=4)
            
            logger.info("=== GUARDANDO CHECKPOINT DESPUES DE TRENDS ===")
            save_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_trends)
            gc.collect()

        # ACCELERATION
        if not (skip_to_momentum or skip_to_streaks or skip_to_time_since):
            logger.info("=== CREANDO ACCELERATION ===")
            vars_criticas_accel = [
                'mrentabilidad', 'mcuentas_saldo', 'mconsumototal_tc', 'cproductos',
                'transacciones_digitales_total', 'payroll_monto_total'
            ]
            conn = create_acceleration_features(conn, SQL_TABLE_NAME, vars_criticas_accel)
            
            logger.info("=== GUARDANDO CHECKPOINT DESPUES DE ACCELERATION ===")
            save_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_accel)
            gc.collect()

        # MOMENTUM
        if not (skip_to_streaks or skip_to_time_since):
            logger.info("=== CREANDO MOMENTUM ===")
            vars_momentum = [
                'mrentabilidad', 'mcuentas_saldo', 'mconsumototal_tc',
                'transacciones_digitales_total', 'inversiones_monto_total',
                'payroll_monto_total', 'cproductos'
            ]
            conn = create_momentum_features(conn, SQL_TABLE_NAME, vars_momentum, recent_window=2, past_start=2, past_end=5)
            
            logger.info("=== GUARDANDO CHECKPOINT DESPUES DE MOMENTUM ===")
            save_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_momentum)
            gc.collect()

        # STREAKS
        if not skip_to_time_since:
            logger.info("=== CREANDO STREAKS ===")
            conditions_streaks = [
                'flag_rentabilidad_negativa = 1',
                'flag_saldo_decreciente = 1',
                'flag_sin_consumo_tc = 1',
                'flag_sin_transacciones_digital = 1',
                'flag_inactivo = 1',
                'flag_perdiendo_productos = 1',
                'flag_sin_payroll = 1',
                'flag_tc_en_cierre = 1',
                'flag_endeudamiento_creciente = 1',
                'flag_desinvirtiendo = 1'
            ]
            output_names_streaks = [
                'streak_rentabilidad_negativa_3m',
                'streak_saldo_decreciente_3m',
                'streak_sin_consumo_tc_3m',
                'streak_sin_transacciones_digital_3m',
                'streak_inactivo_3m',
                'streak_perdiendo_productos_3m',
                'streak_sin_payroll_3m',
                'streak_tc_en_cierre_3m',
                'streak_endeudamiento_creciente_3m',
                'streak_desinvirtiendo_3m'
            ]
            conn = create_streak_features(conn, SQL_TABLE_NAME, conditions_streaks, output_names_streaks, window=3)
            
            logger.info("=== GUARDANDO CHECKPOINT DESPUES DE STREAKS ===")
            save_checkpoint(conn, SQL_TABLE_NAME, checkpoint_after_streaks)
            gc.collect()

        # TIME_SINCE
        logger.info("=== CREANDO TIME_SINCE ===")
        conditions_time_since = [
            'flag_consumo_tc = 1',
            'flag_trx_digital = 1',
            'flag_payroll = 1',
            'flag_plazo_fijo = 1',
            'flag_inversion = 1',
            'flag_perdiendo_productos = 1',
            'flag_pico_saldo = 1',
            'flag_pico_productos = 1',
            'flag_pico_rentabilidad = 1',
            'flag_cambio_status_tc = 1'
        ]
        output_names_time_since = [
            'meses_desde_ultimo_consumo_tc',
            'meses_desde_ultima_trx_digital',
            'meses_desde_ultimo_payroll',
            'meses_desde_ultimo_plazo_fijo',
            'meses_desde_ultima_inversion',
            'meses_desde_cancelacion_producto',
            'meses_desde_pico_saldo',
            'meses_desde_pico_productos',
            'meses_desde_pico_rentabilidad',
            'meses_desde_cambio_status_tc'
        ]
        conn = create_time_since_features(conn, SQL_TABLE_NAME, conditions_time_since, output_names_time_since)

        # GUARDAR RESULTADO FINAL
        logger.info("PASO FINAL: Guardando dataset sin targets...")
        output_sin_targets = OUTPUT_PATH_FE.replace('.parquet', '_sin_targets.parquet')
        
        try:
            conn.execute(f"""
                COPY {SQL_TABLE_NAME} 
                TO '{output_sin_targets}' 
                (FORMAT PARQUET, COMPRESSION UNCOMPRESSED, ROW_GROUP_SIZE 100000)
            """)
            logger.info(f"Dataset guardado exitosamente en: {output_sin_targets}")
        except Exception as e:
            logger.error(f"Error guardando dataset: {e}")
            logger.info("Intentando guardar con particiones por foto_mes...")
            base_path = output_sin_targets.replace('.parquet', '')
            conn.execute(f"""
                COPY {SQL_TABLE_NAME} 
                TO '{base_path}' 
                (FORMAT PARQUET, PARTITION_BY (foto_mes), COMPRESSION ZSTD)
            """)
            logger.info(f"Dataset guardado particionado en: {base_path}/foto_mes=*/")
        
        logger.info("=== FEATURE ENGINEERING COMPLETADO ===")

    except Exception as e:
        logger.error(f"Error durante la ejecución del pipeline: {e}")
        raise

    finally:
        if conn:
            conn.close()
            logger.info("Conexión a DuckDB cerrada.")
        
        if temp_dir:
            cleanup_temp_dir(temp_dir)

if __name__ == "__main__":
    main()