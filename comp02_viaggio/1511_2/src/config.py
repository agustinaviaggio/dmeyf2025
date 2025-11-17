import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Ruta del archivo de configuracion
PATH_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conf.yaml")

try:
    with open(PATH_CONFIG, "r") as f:
        _cfgGeneral = yaml.safe_load(f)
        _cfg = _cfgGeneral["configuracion"]

        STUDY_NAME = _cfgGeneral["STUDY_NAME"]
        DATA_PATH_OPT = _cfg["DATA_PATH_OPT"]
        DATA_PATH_FE = _cfg["DATA_PATH_FE"]
        OUTPUT_PATH_FE = _cfg["OUTPUT_PATH_FE"]
        BUCKET_NAME = _cfg["BUCKET_NAME"]
        SQL_TABLE_NAME = _cfg['SQL_TABLE_NAME']
        SEMILLAS = _cfg["SEMILLAS"]
        
        # Períodos específicos por clase
        PERIODOS_CLASE_1 = _cfg["PERIODOS_CLASE_1"]
        PERIODOS_CLASE_0 = _cfg["PERIODOS_CLASE_0"]
        
        # Períodos de test
        MES_TEST_1 = _cfg["MES_TEST_1"]
        MES_TEST_2 = _cfg["MES_TEST_2"]
        
        # Ganancias
        GANANCIA_ACIERTO = _cfg["GANANCIA_ACIERTO"]
        COSTO_ESTIMULO = _cfg["COSTO_ESTIMULO"]
        
        # Cross Validation
        N_SPLITS = _cfg.get("N_SPLITS", 5)
        UNDERSAMPLING_RATIO = _cfg.get("UNDERSAMPLING_RATIO", 0.20)
        
        # Legacy (mantener por compatibilidad)
        PERIODOS_TRAIN = _cfg.get("PERIODOS_TRAIN", PERIODOS_CLASE_1)
        MESES_TRAIN_BAJA = _cfg.get("MESES_TRAIN_BAJA", ["202101","202102","202103"])
        MESES_TRAIN_CONTINUA = _cfg.get("MESES_TRAIN_CONTINUA", ["202103"])
        MES_VALIDACION = _cfg.get("MES_VALIDACION", ["202104"])
        FINAL_TRAIN = _cfg.get("FINAL_TRAIN", ["202008","202010", "202011", "202012", "202101","202102"])
        FINAL_PREDICT = _cfg.get("FINAL_PREDICT", "202104")

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise