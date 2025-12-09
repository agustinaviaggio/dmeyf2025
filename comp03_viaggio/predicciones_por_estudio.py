import duckdb
import logging
import numpy as np
import pandas as pd
import subprocess
import tempfile
import lightgbm as lgb
import os
import gc
import yaml
import json
from pathlib import Path
from datetime import datetime

# Configuración
ESTUDIOS = ['2511_2', '2611_2', '2711_2']
BUCKET_NAME = "gs://sra_electron_bukito3/"
MES_PREDICCION = 202109

# Para estudios sin metadata
FEATURES_FROM_MODEL = {}

# Logging
os.makedirs("logs_pred_por_estudio", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs_pred_por_estudio/log_{fecha}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def refrescar_credenciales_gcs():
    from google.auth import default
    from google.auth.transport.requests import Request
    credentials, project = default()
    credentials.refresh(Request())
    os.environ['CLOUDSDK_AUTH_ACCESS_TOKEN'] = credentials.token
    return credentials.token


def cargar_metadata_features(study_name):
    """Devuelve lista exacta de features o None."""
    pattern = f"{BUCKET_NAME}resultados/metadata_features_{study_name}_*.json"
    result = subprocess.run(['gsutil', 'ls', pattern],
                            capture_output=True, text=True)

    if result.returncode != 0:
        return None

    archivos = result.stdout.strip().split("\n")
    if not archivos or archivos == ['']:
        return None

    archivo = sorted(archivos)[-1]

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(['gsutil', 'cp', archivo, tmp_path], check=True)

    with open(tmp_path, "r") as f:
        data = json.load(f)

    os.unlink(tmp_path)
    return data["features"]


def extraer_features_del_modelo(modelo, feature_cols):
    """Para estudios sin metadata: tomamos feature_name() del modelo."""
    cols = modelo.feature_name()
    cols = [c for c in cols if c not in ['target_binario', 'target_ternario', 'foto_mes']]

    faltan = [c for c in cols if c not in feature_cols]
    if faltan:
        logger.error("Faltan columnas en el parquet!")
        raise ValueError(faltan)

    return cols


def cargar_datos():
    logger.info("Cargando datos del parquet...")

    conf_path = Path(f"~/sky_dmeyf/{ESTUDIOS[0]}/conf.yaml").expanduser()
    config = yaml.safe_load(open(conf_path))["configuracion"]
    data_path = config["DATA_PATH_OPT"]

    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    token = refrescar_credenciales_gcs()
    conn.execute(f"""
        CREATE SECRET (TYPE GCS, PROVIDER config, BEARER_TOKEN '{token}')
    """)

    conn.execute(f"""
        CREATE TABLE datos AS SELECT * FROM read_parquet('{data_path}')
    """)

    data = conn.execute(f"SELECT * FROM datos WHERE foto_mes = {MES_PREDICCION}").fetchnumpy()
    conn.close()

    columnas_prohibidas = {'foto_mes', 'target_binario', 'target_ternario'}
    feature_cols = [c for c in data.keys() if c not in columnas_prohibidas]

    X = np.column_stack([data[c] for c in feature_cols])
    n_clientes = data["numero_de_cliente"]

    return X, n_clientes, feature_cols


def predecir_estudio(study, modelos, X, feature_cols):
    logger.info(f"Procesando features para {study}...")

    metadata = cargar_metadata_features(study)

    if metadata is None:
        if study not in FEATURES_FROM_MODEL:
            FEATURES_FROM_MODEL[study] = extraer_features_del_modelo(modelos[0], feature_cols)
        features_ok = FEATURES_FROM_MODEL[study]
    else:
        features_ok = metadata

    indices = [feature_cols.index(f) for f in features_ok]
    X_f = X[:, indices]

    preds = [m.predict(X_f) for m in modelos]
    return np.mean(preds, axis=0)


def descargar_modelos(study):
    logger.info(f"Descargando modelos de {study}...")
    refrescar_credenciales_gcs()

    tmp_dir = Path(f"/tmp/{study}_modelos")
    tmp_dir.mkdir(exist_ok=True)

    pattern = f"{BUCKET_NAME}modelos_finales/{study}_seed_*.txt"
    subprocess.run(['gsutil', '-m', 'cp', pattern, str(tmp_dir)], check=True)

    modelos = []
    for archivo in sorted(tmp_dir.glob("*.txt")):
        modelos.append(lgb.Booster(model_file=str(archivo)))

    import shutil
    shutil.rmtree(tmp_dir)
    return modelos


def main():
    X, clientes, feature_cols = cargar_datos()

    os.makedirs("preds_por_estudio", exist_ok=True)

    for study in ESTUDIOS:
        logger.info("=" * 60)
        logger.info(f"ESTUDIO {study}")
        logger.info("=" * 60)

        modelos = descargar_modelos(study)
        pred = predecir_estudio(study, modelos, X, feature_cols)

        df = pd.DataFrame({"numero_de_cliente": clientes, "prob": pred})
        out = f"preds_por_estudio/preds_{study}_{fecha}.csv"
        df.to_csv(out, index=False)
        logger.info(f"✓ Guardado {out}")

        # Subir al bucket
        subprocess.run(["gsutil", "cp", out, BUCKET_NAME + "preds_por_estudio/"])


if __name__ == "__main__":
    main()
