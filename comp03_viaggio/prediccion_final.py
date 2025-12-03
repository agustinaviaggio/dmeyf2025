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
THRESHOLD = 0.76
ENVIOS_FIJOS = 11000

# Logging
os.makedirs("logs", exist_ok=True)
fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_prediccion_final_{fecha}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def refrescar_credenciales_gcs():
    """Refresca las credenciales de GCS."""
    try:
        from google.auth import default
        from google.auth.transport.requests import Request
        
        credentials, project = default()
        credentials.refresh(Request())
        
        os.environ['CLOUDSDK_AUTH_ACCESS_TOKEN'] = credentials.token
        return credentials.token
    except Exception as e:
        logger.error(f"Error refrescando credenciales: {e}")
        raise


def cargar_features_por_estudio(study_name, bucket_name):
    """Carga las features que usó un estudiO."""
    
    # Primero intentar cargar metadata de features
    try:
        refrescar_credenciales_gcs()
        
        gcs_pattern = f"{bucket_name}resultados/metadata_features_{study_name}_*.json"
        result = subprocess.run(
            ['gsutil', 'ls', gcs_pattern],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            archivos = result.stdout.strip().split('\n')
            if archivos and archivos[0]:
                archivo = sorted(archivos)[-1]
                logger.info(f"  Encontrado metadata: {archivo.split('/')[-1]}")
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    subprocess.run(
                        ['gsutil', 'cp', archivo, tmp_path],
                        check=True,
                        timeout=60,
                        capture_output=True
                    )
                    
                    with open(tmp_path, 'r') as f:
                        metadata = json.load(f)
                    
                    features = metadata['features']
                    logger.info(f"  ✓ Cargadas {len(features)} features específicas del estudio")
                    return features
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
    except Exception as e:
        logger.info(f"  No se encontraron features específicas ({e})")
    
    logger.info(f"  Se usarán todas las features disponibles")
    return None


def descargar_modelos_estudio(study_name, bucket_name):
    """Descarga los 50 modelos de un estudio desde GCS."""
    logger.info(f"\nDescargando modelos de {study_name}...")
    
    refrescar_credenciales_gcs()
    
    local_dir = Path.home() / f"modelos_temp_{study_name}"
    local_dir.mkdir(exist_ok=True)
    
    # Descargar todos los modelos del estudio
    gcs_pattern = f"{bucket_name}modelos_finales/{study_name}_seed_*.txt"
    
    result = subprocess.run(
        ['gsutil', '-m', 'cp', gcs_pattern, str(local_dir)],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode != 0:
        logger.error(f"Error descargando modelos: {result.stderr}")
        raise Exception(f"No se pudieron descargar modelos de {study_name}")
    
    # Cargar modelos
    modelos = []
    archivos = sorted(local_dir.glob(f"{study_name}_seed_*.txt"))
    
    logger.info(f"  Encontrados {len(archivos)} modelos")
    
    for archivo in archivos:
        modelo = lgb.Booster(model_file=str(archivo))
        modelos.append(modelo)
    
    logger.info(f"  ✓ {len(modelos)} modelos cargados")
    
    # Limpiar archivos
    import shutil
    shutil.rmtree(local_dir)
    
    return modelos


def cargar_datos_prediccion(mes_prediccion):
    """Carga datos del mes de predicción desde GCS."""
    logger.info(f"\nCargando datos de {mes_prediccion}...")
    
    # Cargar config del primer estudio para obtener data_path
    conf_file = Path(f"~/sky_dmeyf/{ESTUDIOS[0]}/conf.yaml").expanduser()
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)['configuracion']
    
    data_path = config['DATA_PATH_OPT']
    logger.info(f"  Ruta: {data_path}")
    
    # Conectar DuckDB
    conn = duckdb.connect(database=':memory:')
    conn.execute("SET temp_directory='/tmp'")
    
    # Configurar GCS
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
    
    logger.info(f"  Creando tabla desde parquet...")
    conn.execute(f"""
        CREATE TABLE datos AS 
        SELECT *
        FROM read_parquet('{data_path}')
    """)
    
    logger.info(f"  ✓ Tabla creada")
    logger.info(f"  Filtrando foto_mes={mes_prediccion}...")
    
    # Consultar el mes específico
    query = f"SELECT * FROM datos WHERE foto_mes = {mes_prediccion}"
    data = conn.execute(query).fetchnumpy()
    
    # Obtener numero_de_cliente
    if 'numero_de_cliente' in data:
        numeros_cliente = data['numero_de_cliente']
    else:
        logger.warning("No se encontró 'numero_de_cliente', usando índices")
        numeros_cliente = np.arange(len(list(data.values())[0]))

    feature_cols = [col for col in data.keys() 
                   if col not in ['target_binario', 'target_ternario', 'foto_mes']]
    
    X = np.column_stack([data[col] for col in feature_cols])
    
    logger.info(f"  ✓ {len(numeros_cliente):,} registros, {len(feature_cols)} features")
    
    conn.close()
    
    return X, numeros_cliente, feature_cols


def predecir_ensemble_estudio(modelos, X, feature_cols, study_name, bucket_name):
    """Predice con un ensemble de modelos."""
    logger.info(f"  Verificando features del estudio...")
    
    # Cargar features específicas del estudio
    features_estudio = cargar_features_por_estudio(study_name, bucket_name)
    
    if features_estudio is not None:
        # Filtrar X solo con las features que usó este estudio
        indices_features = [i for i, col in enumerate(feature_cols) if col in features_estudio]
        X_filtrado = X[:, indices_features]
        logger.info(f"  Usando {len(indices_features)} features del estudio")
    else:
        X_filtrado = X
        logger.info(f"  Usando todas las features ({X.shape[1]})")
    
    logger.info(f"  Prediciendo con {len(modelos)} modelos...")
    
    predicciones = []
    for modelo in modelos:
        pred = modelo.predict(X_filtrado)
        predicciones.append(pred)
        gc.collect()
    
    # Promedio de todos los modelos
    ensemble_pred = np.mean(predicciones, axis=0)
    
    return ensemble_pred


def generar_submission(probabilidades, numeros_cliente, threshold, n_envios, nombre_archivo):
    """Genera archivo de submission con los clientes a enviar."""
    
    # Ordenar por probabilidad descendente
    indices_ordenados = np.argsort(probabilidades)[::-1]
    
    if threshold is not None:
        seleccionados = probabilidades >= threshold
        indices_seleccionados = np.where(seleccionados)[0]
        indices_seleccionados = indices_seleccionados[np.argsort(probabilidades[seleccionados])[::-1]]
    else:
        indices_seleccionados = indices_ordenados[:n_envios]
    
    clientes_seleccionados = numeros_cliente[indices_seleccionados]
    
    # Crear DataFrame
    df = pd.DataFrame({
        'numero_de_cliente': clientes_seleccionados
    })
    
    # Guardar
    output_dir = Path("submissions")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / nombre_archivo
    df.to_csv(output_file, index=False)
    
    logger.info(f"  ✓ Generado: {output_file}")
    logger.info(f"    Clientes seleccionados: {len(clientes_seleccionados):,}")
    
    # Subir a GCS
    gcs_path = f"{BUCKET_NAME}submissions/{nombre_archivo}"
    subprocess.run(['gsutil', 'cp', str(output_file), gcs_path])
    
    logger.info(f"  ✓ Subido a: {gcs_path}")
    
    return output_file


def main():
    logger.info("="*80)
    logger.info("PREDICCIÓN FINAL PARA SUBMISSION")
    logger.info("="*80)
    logger.info(f"Mes de predicción: {MES_PREDICCION}")
    logger.info(f"Estudios: {ESTUDIOS}")
    logger.info(f"Total modelos a cargar: {len(ESTUDIOS) * 50} = 150")
    
    # 1. Cargar datos de predicción
    X, numeros_cliente, feature_cols = cargar_datos_prediccion(MES_PREDICCION)
    
    # 2. Cargar modelos y predecir por estudio
    predicciones_estudios = []
    
    for study_name in ESTUDIOS:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESANDO ESTUDIO: {study_name}")
        logger.info(f"{'='*80}")
        
        # Descargar y cargar modelos
        modelos = descargar_modelos_estudio(study_name, BUCKET_NAME)
        
        # Predecir ensemble del estudio
        pred_ensemble = predecir_ensemble_estudio(
            modelos, 
            X, 
            feature_cols,
            study_name,
            BUCKET_NAME
        )
        predicciones_estudios.append(pred_ensemble)
        
        logger.info(f"  ✓ Predicciones completadas para {study_name}")
        
        # Liberar memoria
        del modelos
        gc.collect()
    
    # 3. Combinar predicciones (promedio simple de los 3 estudios)
    logger.info("\n" + "="*80)
    logger.info("COMBINANDO PREDICCIONES")
    logger.info("="*80)
    logger.info("Estrategia: Promedio simple de 3 estudios")
    
    predicciones_finales = np.mean(predicciones_estudios, axis=0)
    
    logger.info(f"  ✓ Predicciones finales generadas")
    logger.info(f"  Min prob: {predicciones_finales.min():.6f}")
    logger.info(f"  Max prob: {predicciones_finales.max():.6f}")
    logger.info(f"  Media prob: {predicciones_finales.mean():.6f}")
    
    # 4. Generar submissions
    logger.info("\n" + "="*80)
    logger.info("GENERANDO ARCHIVOS DE SUBMISSION")
    logger.info("="*80)
    
    # Submission 1: Con threshold
    logger.info(f"\n1. SUBMISSION CON THRESHOLD = {THRESHOLD}")
    archivo_threshold = generar_submission(
        predicciones_finales,
        numeros_cliente,
        threshold=THRESHOLD,
        n_envios=None,
        nombre_archivo=f"submission_threshold_{THRESHOLD:.6f}_{fecha}.csv"
    )
    
    # Submission 2: Con envíos fijos
    logger.info(f"\n2. SUBMISSION CON ENVÍOS FIJOS = {ENVIOS_FIJOS}")
    archivo_fijos = generar_submission(
        predicciones_finales,
        numeros_cliente,
        threshold=None,
        n_envios=ENVIOS_FIJOS,
        nombre_archivo=f"submission_envios_{ENVIOS_FIJOS}_{fecha}.csv"
    )
    
    # Resumen final
    logger.info("\n" + "="*80)
    logger.info("PREDICCIÓN COMPLETADA")
    logger.info("="*80)
    logger.info(f"\nArchivos generados:")
    logger.info(f"  1. {archivo_threshold}")
    logger.info(f"  2. {archivo_fijos}")
    logger.info(f"\nSubidos a: {BUCKET_NAME}submissions/")
    logger.info("\n✓ Pipeline de predicción completado exitosamente")


if __name__ == "__main__":
    main()