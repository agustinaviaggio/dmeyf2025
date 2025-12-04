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

# -------------------------
# CONFIGURACIÓN
# -------------------------
ESTUDIOS = ['2511_2', '2611_2', '2711_2']
BUCKET_NAME = "gs://sra_electron_bukito3/"
MES_PREDICCION = 202109
ENVIOS_FIJOS = 11000

# -------------------------
# LOGGING
# -------------------------
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_prediccion_por_estudio_topk_{fecha}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FEATURES_FROM_MODEL = {}

# -------------------------
# UTILS
# -------------------------
def refrescar_credenciales_gcs():
    from google.auth import default
    from google.auth.transport.requests import Request
    
    credentials, project = default()
    credentials.refresh(Request())
    os.environ['CLOUDSDK_AUTH_ACCESS_TOKEN'] = credentials.token
    return credentials.token


def cargar_features_por_estudio(study_name, bucket_name):
    """Carga el JSON de features si existe. Si no, se usan las del modelo."""
    try:
        refrescar_credenciales_gcs()
        gcs_pattern = f"{bucket_name}resultados/metadata_features_{study_name}_*.json"
        result = subprocess.run(['gsutil', 'ls', gcs_pattern],
                                capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            archivos = result.stdout.strip().split('\n')
            if archivos and archivos[0]:
                archivo = sorted(archivos)[-1]
                logger.info(f"  Metadata encontrada: {archivo.split('/')[-1]}")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    tmp_path = tmp.name
                subprocess.run(['gsutil', 'cp', archivo, tmp_path],
                               check=True, timeout=60, capture_output=True)

                with open(tmp_path, 'r') as f:
                    metadata = json.load(f)

                os.unlink(tmp_path)
                return metadata["features"]
    except:
        pass

    logger.info("  No hay metadata JSON → se usarán features del modelo.")
    return None


def descargar_modelos_estudio(study_name, bucket_name):
    logger.info(f"Descargando modelos de {study_name}...")
    refrescar_credenciales_gcs()

    local_dir = Path.home() / f"modelos_temp_{study_name}"
    local_dir.mkdir(exist_ok=True)

    gcs_pattern = f"{bucket_name}modelos_finales/{study_name}_seed_*.txt"
    result = subprocess.run(['gsutil', '-m', 'cp', gcs_pattern, str(local_dir)],
                            capture_output=True, text=True, timeout=300)

    archivos = sorted(local_dir.glob(f"{study_name}_seed_*.txt"))
    logger.info(f"  {len(archivos)} modelos encontrados.")

    modelos = [lgb.Booster(model_file=str(a)) for a in archivos]

    import shutil
    shutil.rmtree(local_dir)
    return modelos


def cargar_datos_prediccion(mes_prediccion):
    logger.info(f"Cargando datos {mes_prediccion}...")

    conf_file = Path(f"~/sky_dmeyf/{ESTUDIOS[0]}/conf.yaml").expanduser()
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)['configuracion']

    data_path = config['DATA_PATH_OPT']
    logger.info(f"  Ruta: {data_path}")

    conn = duckdb.connect(database=':memory:')
    conn.execute("SET temp_directory='/tmp'")

    # Autenticación GCS
    token = refrescar_credenciales_gcs()
    conn.execute("INSTALL httpfs;")
    conn.execute("LOAD httpfs;")
    conn.execute(f"""
        CREATE SECRET (
            TYPE GCS,
            PROVIDER config,
            BEARER_TOKEN '{token}'
        )
    """)

    conn.execute(f"""
        CREATE TABLE datos AS 
        SELECT * FROM read_parquet('{data_path}')
    """)

    data = conn.execute(
        f"SELECT * FROM datos WHERE foto_mes = {mes_prediccion}"
    ).fetchnumpy()

    numeros_cliente = data["numero_de_cliente"]

    columnas_prohibidas = {"target_binario", "target_ternario", "foto_mes"}
    feature_cols = [c for c in data.keys() if c not in columnas_prohibidas]

    X = np.column_stack([data[c] for c in feature_cols])
    conn.close()

    logger.info(f"  {len(X)} registros, {len(feature_cols)} features.")

    return X, numeros_cliente, feature_cols


def obtener_features_modelo(modelo, feature_cols, study_name):
    names = modelo.feature_name()
    columnas_prohibidas = {"target_binario", "target_ternario", "foto_mes"}
    names = [f for f in names if f not in columnas_prohibidas]

    faltantes = [f for f in names if f not in feature_cols]
    if faltantes:
        logger.error(f"ERROR: Columnas faltantes para {study_name}: {faltantes[:10]}")
        raise ValueError("Faltan columnas en predicción.")

    return names


def predecir_estudio(modelos, X, feature_cols, study_name):
    feats_json = cargar_features_por_estudio(study_name, BUCKET_NAME)

    if feats_json is None:
        feat_list = obtener_features_modelo(modelos[0], feature_cols, study_name)
    else:
        feat_list = feats_json

    idx = [feature_cols.index(f) for f in feat_list]
    X_filtrado = X[:, idx]

    preds = np.mean([m.predict(X_filtrado) for m in modelos], axis=0)

    logger.info(f"  Stats {study_name}: min={preds.min():.5f}, max={preds.max():.5f}, mean={preds.mean():.5f}")
    return preds


def generar_submission_topk(prob, clientes, study_name):
    orden = np.argsort(prob)[::-1][:ENVIOS_FIJOS]
    df = pd.DataFrame({"numero_de_cliente": clientes[orden]})

    os.makedirs("submissions_topk", exist_ok=True)
    filename = f"submission_{study_name}_top{ENVIOS_FIJOS}_{fecha}.csv"
    path = Path("submissions_topk") / filename

    df.to_csv(path, index=False)

    logger.info(f"  ✓ Submission topk guardado: {path}")
    subprocess.run(["gsutil", "cp", str(path), f"{BUCKET_NAME}submissions/{filename}"])

    return path


# -------------------------
# MAIN
# -------------------------
def main():
    logger.info("="*80)
    logger.info("PREDICCIONES INDIVIDUALES (SOLO TOP 11000)")
    logger.info("="*80)

    X, clientes, feature_cols = cargar_datos_prediccion(MES_PREDICCION)

    for study in ESTUDIOS:
        logger.info(f"\n===== ESTUDIO {study} =====")

        modelos = descargar_modelos_estudio(study, BUCKET_NAME)
        preds = predecir_estudio(modelos, X, feature_cols, study)

        generar_submission_topk(preds, clientes, study)

        del modelos
        gc.collect()


if __name__ == "__main__":
    main()
