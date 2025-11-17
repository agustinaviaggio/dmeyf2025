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
        
        # Períodos para Time Series CV
        PERIODOS_TRAIN = _cfg.get("PERIODOS_TRAIN", ["202101", "202102", "202103", "202104"])
        
        # Legacy (mantener por compatibilidad con código viejo)
        MESES_TRAIN_BAJA = _cfg.get("MESES_TRAIN_BAJA", ["202101","202102","202103"])
        MESES_TRAIN_CONTINUA = _cfg.get("MESES_TRAIN_CONTINUA", ["202103"])
        MES_VALIDACION = _cfg.get("MES_VALIDACION", ["202104"])
        
        MES_TEST_1 = _cfg["MES_TEST_1"]
        MES_TEST_2 = _cfg["MES_TEST_2"]
        GANANCIA_ACIERTO = _cfg["GANANCIA_ACIERTO"]
        COSTO_ESTIMULO = _cfg["COSTO_ESTIMULO"]
        FINAL_TRAIN = _cfg["FINAL_TRAIN"]
        FINAL_PREDICT = _cfg["FINAL_PREDICT"]
        
        # Time Series CV
        N_SPLITS = _cfg.get("N_SPLITS", 3)
        CV_STRATEGY = _cfg.get("CV_STRATEGY", "expanding")
        MIN_TRAIN_SIZE = _cfg.get("MIN_TRAIN_SIZE", 2)
        VALIDATION_SIZE = _cfg.get("VALIDATION_SIZE", 1)
        GAP = _cfg.get("GAP", 0)
        UNDERSAMPLING_RATIO = _cfg.get("UNDERSAMPLING_RATIO", 0.01)

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise