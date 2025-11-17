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

        # 6. Borrar atributos individuales usandos para crear los atributos tipo suma para las tarjetas de crédito
        conn = drop_columns(conn, SQL_TABLE_NAME, cols_visa_str+cols_master_str)

        # 7. Crear atributos tipo ratio entre pares de variables m_ y c_
        conn = create_ratio_m_c_attributes(conn, SQL_TABLE_NAME)

        # 13. Deuda total en préstamos
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('mprestamos_personales', 'mprestamos_prendarios', 'mprestamos_hipotecarios'),
                ('cprestamos_personales', 'cprestamos_prendarios', 'cprestamos_hipotecarios')
            ],
            output_names=['deuda_total_prestamos', 'prestamos_count_total']
        )

        # 14. Endeudamiento total (TC + préstamos)
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('msaldototal_tc', 'deuda_total_prestamos')
            ],
            output_names=['endeudamiento_total']
        )

        # 15. Inversiones total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('mplazo_fijo_dolares', 'mplazo_fijo_pesos'),
                ('mplazo_fijo_dolares', 'mplazo_fijo_pesos', 'minversion1_pesos', 'minversion1_dolares', 'minversion2'),
                ('cplazo_fijo', 'cinversion1', 'cinversion2')
            ],
            output_names=['plazo_fijo_total', 'inversiones_monto_total', 'inversiones_count_total']
        )

        # 16. Seguros total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('cseguro_vida', 'cseguro_auto', 'cseguro_vivienda', 'cseguro_accidentes_personales')
            ],
            output_names=['seguros_total']
        )

        # 17. Payroll total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('mpayroll', 'mpayroll2'),
                ('cpayroll_trx', 'cpayroll2_trx')
            ],
            output_names=['payroll_monto_total', 'payroll_trx_total']
        )

        # 19. Pagos de servicios total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('cpagodeservicios', 'cpagomiscuentas'),
                ('mpagodeservicios', 'mpagomiscuentas')
            ],
            output_names=['pagos_servicios_count_total', 'pagos_servicios_monto_total']
        )

        # 13.9 Comisiones total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('ccomisiones_mantenimiento', 'ccomisiones_otras'),
                ('mcomisiones_mantenimiento', 'mcomisiones_otras')
            ],
            output_names=['comisiones_count_total', 'comisiones_monto_total']
        )

        # 13.11 Forex balance (diferencia)
        conn = create_diff_features(
            conn, SQL_TABLE_NAME,
            column_pairs=[('mforex_buy', 'mforex_sell')],
            output_names=['forex_balance']
        )

         # 13.13 Transferencias balance
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

        # 13.16 Canales físicos
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('ccajas_transacciones', 'ccajas_consultas', 'ccajas_depositos', 'ccajas_extracciones', 'ccajas_otras')
            ],
            output_names=['transacciones_cajas_total']
        )

        # 13.17 ATM total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('catm_trx', 'catm_trx_other'),
                ('matm', 'matm_other')
            ],
            output_names=['atm_trx_total', 'atm_monto_total']
        )

        # 13.18 Transacciones totales
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('transacciones_digitales_total', 'ccallcenter_transacciones', 'transacciones_cajas_total', 'atm_trx_total')
            ],
            output_names=['transacciones_totales']
        )

        # 13.19 Cuentas total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('ccuenta_corriente', 'ccaja_ahorro'),
                ('mcuenta_corriente', 'mcuenta_corriente_adicional', 'mcaja_ahorro', 'mcaja_ahorro_adicional')
            ],
            output_names=['cuentas_total', 'saldo_pesos_total']
        )

        # 13.20 Margen total
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('mactivos_margen', 'mpasivos_margen')
            ],
            output_names=['margen_total']
        )

        # 13.22 Activos totales y patrimonio
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

        # 13.23 Actividad por tipo
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('ctransferencias_emitidas', 'ctransferencias_recibidas', 'cextraccion_autoservicio', 'pagos_servicios_count_total'),
                ('cplazo_fijo', 'cinversion1', 'cinversion2', 'cforex')
            ],
            output_names=['actividad_transaccional', 'actividad_inversora']
        )

        # 13.24 FLAGS de productos
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

        # 13.25 Diversificación de productos
        conn = create_sum_features(
            conn, SQL_TABLE_NAME,
            columns_to_sum=[
                ('tiene_cuenta_corriente', 'tiene_caja_ahorro', 'tiene_tc_activa', 'tiene_prestamos', 'tiene_inversiones', 'tiene_seguros')
            ],
            output_names=['diversificacion_productos']
        )

        # 14.1 Ratios de endeudamiento
        conn = create_ratio_features(
            conn, SQL_TABLE_NAME,
            numerator_cols=['endeudamiento_total', 'msaldototal_tc', 'mconsumototal_tc'],
            denominator_cols=['mcuentas_saldo', 'mlimitecompra_tc', 'mlimitecompra_tc'],
            output_names=['ratio_endeudamiento_vs_saldo', 'ratio_saldo_tc_vs_limite', 'ratio_consumo_tc_vs_limite']
        )

        # 14.2 Ratios de comportamiento de pago
        conn = create_ratio_features(
            conn, SQL_TABLE_NAME,
            numerator_cols=['mpagado_tc', 'mpagado_tc', 'madelantopesos_tc'],
            denominator_cols=['mconsumototal_tc', 'mpagominimo_tc', 'mconsumototal_tc'],
            output_names=['ratio_pagos_vs_consumo_tc', 'ratio_pagado_vs_minimo_tc', 'ratio_adelantos_vs_consumo_tc']
        )

        # 14.3 Ratios de inversión
        conn = create_ratio_features(
            conn, SQL_TABLE_NAME,
            numerator_cols=['inversiones_monto_total', 'inversiones_monto_total'],
            denominator_cols=['mcuentas_saldo', 'activos_totales'],
            output_names=['ratio_inversiones_vs_saldo', 'ratio_inversiones_vs_activos']
        )

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
        '''
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
        '''

        conn = create_all_window_attributes(conn, SQL_TABLE_NAME, excluir_columnas_max, month_window = 3)

        conn = create_behavioral_flags(conn, SQL_TABLE_NAME)
        conn = create_active_quarter_feature(conn, SQL_TABLE_NAME)

        
        # Variables críticas para tendencias
        vars_criticas_trend = [
            'mrentabilidad', 'mcuentas_saldo', 'mconsumototal_tc', 'cproductos',
            'transacciones_digitales_total', 'payroll_monto_total', 'inversiones_monto_total',
            'endeudamiento_total', 'margen_total',
            'cuentas_total', 'seguros_total', 'diversificacion_productos',
            'actividad_transaccional', 'actividad_inversora', 'comisiones_monto_total'
        ]

        # También tendencias para los ratios más importantes
        ratios_trend = [
            'ratio_endeudamiento_vs_saldo', 'ratio_consumo_tc_vs_limite', 
            'ratio_pagos_vs_consumo_tc', 'ratio_inversiones_vs_saldo'
        ]

        conn = create_trend_features(conn, SQL_TABLE_NAME, vars_criticas_trend + ratios_trend, window=4)

        
        vars_criticas_accel = [
            'mrentabilidad', 'mcuentas_saldo', 'mconsumototal_tc', 'cproductos',
            'transacciones_digitales_total', 'payroll_monto_total'
        ]

        conn = create_acceleration_features(conn, SQL_TABLE_NAME, vars_criticas_accel)

       
        vars_momentum = [
            'mrentabilidad', 'mcuentas_saldo', 'mconsumototal_tc',
            'transacciones_digitales_total', 'inversiones_monto_total',
            'payroll_monto_total', 'cproductos'
        ]

        conn = create_momentum_features(conn, SQL_TABLE_NAME, vars_momentum, recent_window=2, past_start=2, past_end=5)

        
        logger.info("=== CREANDO STREAKS ===")
        
        conditions_streaks = [
            'flag_rentabilidad_negativa = 1',           # ← Ahora usa flag
            'flag_saldo_decreciente = 1',               # ← Ahora usa flag
            'flag_sin_consumo_tc = 1',                  # ← Ahora usa flag
            'flag_sin_transacciones_digital = 1',       # ← Ahora usa flag
            'flag_inactivo = 1',                  
            'flag_perdiendo_productos = 1',             # ← Ahora usa flag
            'flag_sin_payroll = 1',                     # ← Ahora usa flag
            'flag_tc_en_cierre = 1',                    # ← Ahora usa flag
            'flag_endeudamiento_creciente = 1',         # ← Ahora usa flag
            'flag_desinvirtiendo = 1'                   # ← Ahora usa flag
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

        # ============================================================
        # TIME_SINCE - AHORA USANDO FLAGS PRE-CALCULADOS
        # ============================================================
        
        logger.info("=== CREANDO TIME_SINCE ===")
        
        conditions_time_since = [
            'flag_consumo_tc = 1',                      # ← Ahora usa flag
            'flag_trx_digital = 1',                     # ← Ahora usa flag
            'flag_payroll = 1',                         # ← Ahora usa flag
            'flag_plazo_fijo = 1',                      # ← Ahora usa flag
            'flag_inversion = 1',                       # ← Ahora usa flag
            'flag_perdiendo_productos = 1',             # ← Ahora usa flag (cancelación producto)
            'flag_pico_saldo = 1',                      # ← Ahora usa flag
            'flag_pico_productos = 1',                  # ← Ahora usa flag
            'flag_pico_rentabilidad = 1',               # ← Ahora usa flag
            'flag_cambio_status_tc = 1'                 # ← Ahora usa flag
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

        # ============================================================
        # TARGETS Y GUARDADO (sin cambios)
        # ============================================================
        
        logger.info("=== GENERANDO TARGETS ===")
        conn = generar_targets(conn, SQL_TABLE_NAME)

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