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
        DATA_PATH_ANT = _cfg["DATA_PATH_ANT"]
        DATA_PATH_FE = _cfg["DATA_PATH_FE"]
        OUTPUT_PATH_FE = _cfg["OUTPUT_PATH_FE"]
        BUCKET_NAME = _cfg["BUCKET_NAME"]
        SQL_TABLE_NAME = _cfg['SQL_TABLE_NAME']

except Exception as e:
    logger.error(f"Error al cargar el archivo de configuracion: {e}")
    raise