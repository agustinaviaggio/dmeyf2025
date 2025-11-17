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
        MESES_TRAIN = _cfg["MESES_TRAIN"]
        MES_VALIDACION = _cfg["MES_VALIDACION"]
        MES_TEST_1 = _cfg["MES_TEST_1"]
        MES_TEST_2 = _cfg["MES_TEST_2"]
        GANANCIA_ACIERTO = _cfg["GANANCIA_ACIERTO"]
        COSTO_ESTIMULO = _cfg["COSTO_ESTIMULO"]
        FINAL_TRAIN = _cfg["FINAL_TRAIN"]
        FINAL_PREDICT = _cfg["FINAL_PREDICT"]

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise