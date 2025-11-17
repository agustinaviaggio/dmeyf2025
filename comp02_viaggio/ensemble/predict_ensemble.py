# predict_final_competition.py (VERSIÓN CORREGIDA)
import yaml
import duckdb
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import logging
import os
import gc
from tqdm import tqdm
from collections import defaultdict
from src.features import create_sql_table_from_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CompetitionPredictor:
    def __init__(self, ensemble_config_path: str):
        """
        Predictor para la competencia final.
        
        Args:
            ensemble_config_path: Ruta al YAML de configuración del ensemble
        """
        with open(ensemble_config_path) as f:
            self.ensemble_config = yaml.safe_load(f)
        
        self.output_dir = Path(self.ensemble_config['ensemble']['output_dir']).expanduser()
        self.models_dir = self.output_dir / "trained_models"
        self.final_output_dir = self.output_dir / "final_submission"
        self.final_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Final output: {self.final_output_dir}")
    
    def load_model_config(self, config_dir: str) -> dict:
        """Carga el conf.yaml del modelo"""
        config_path = Path(config_dir).expanduser() / "conf.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró {config_path}")
        
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        
        model_config = {
            'STUDY_NAME': full_config.get('STUDY_NAME'),
            **full_config.get('configuracion', {})
        }
        
        return model_config
    
    def predict_by_study(
        self,
        study_name: str,
        model_files: List[Path],
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        prediction_period: str = "202108"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice con todos los modelos de un study específico.
        
        Returns:
            Tuple[probabilidades_promedio, numero_de_cliente]
        """
        # Query para 202108
        query = f"SELECT * FROM {table_name} WHERE foto_mes = '{prediction_period}'"
        
        logger.info(f"  Cargando datos de {prediction_period}...")
        data = conn.execute(query).fetchnumpy()
        
        # Extraer numero_de_cliente
        numero_de_cliente = data['numero_de_cliente']
        
        logger.info(f"  Total clientes: {len(numero_de_cliente):,}")
        
        # Predecir con cada modelo
        study_predictions = []
        
        for model_file in tqdm(model_files, desc=f"  Prediciendo {study_name}"):
            model_package = joblib.load(model_file)
            model = model_package['model']
            feature_cols = model_package['feature_cols']
            
            # Preparar features
            X = np.column_stack([data[col] for col in feature_cols])
            
            # Predecir
            pred = model.predict(X)
            study_predictions.append(pred)
            
            # Limpiar
            del model, model_package, X
            gc.collect()
        
        # Promedio de este study
        study_proba_avg = np.mean(study_predictions, axis=0)
        
        logger.info(f"  ✓ Predicciones completadas para {study_name}")
        logger.info(f"    Modelos usados: {len(study_predictions)}")
        logger.info(f"    Probabilidad media: {study_proba_avg.mean():.4f}")
        
        # Limpiar
        del data, study_predictions
        gc.collect()
        
        return study_proba_avg, numero_de_cliente
    
    def select_clients_fixed_threshold(
        self,
        probas: np.ndarray,
        numero_de_cliente: np.ndarray,
        n_envios: int = 11000
    ) -> Tuple[np.ndarray, float]:
        """
        Selecciona top N clientes por probabilidad.
        
        Returns:
            Tuple[clientes_seleccionados, umbral_probabilidad]
        """
        # Ordenar por probabilidad descendente
        idx_sorted = np.argsort(probas)[::-1]
        
        # Top N
        idx_selected = idx_sorted[:n_envios]
        clientes_selected = numero_de_cliente[idx_selected]
        umbral = probas[idx_sorted[n_envios-1]]
        
        logger.info(f"\nSelección por top {n_envios}:")
        logger.info(f"  Clientes seleccionados: {len(clientes_selected):,}")
        logger.info(f"  Umbral de probabilidad: {umbral:.6f}")
        logger.info(f"  Probabilidad máxima seleccionada: {probas[idx_sorted[0]]:.6f}")
        logger.info(f"  Probabilidad mínima seleccionada: {umbral:.6f}")
        
        return clientes_selected, umbral
    
    def select_clients_probability_threshold(
        self,
        probas: np.ndarray,
        numero_de_cliente: np.ndarray,
        ganancia_acierto: int = 780000,
        costo_estimulo: int = 20000
    ) -> Tuple[np.ndarray, float, int]:
        """
        Selecciona clientes donde: proba * ganancia > costo.
        
        Umbral óptimo: proba > costo / ganancia
        
        Returns:
            Tuple[clientes_seleccionados, umbral_probabilidad, n_envios]
        """
        # Umbral teórico: punto de equilibrio
        umbral_teorico = costo_estimulo / ganancia_acierto
        
        # Seleccionar clientes sobre el umbral
        mask = probas > umbral_teorico
        clientes_selected = numero_de_cliente[mask]
        
        logger.info(f"\nSelección por umbral de probabilidad:")
        logger.info(f"  Umbral teórico (break-even): {umbral_teorico:.6f}")
        logger.info(f"  Clientes seleccionados: {len(clientes_selected):,}")
        if len(clientes_selected) > 0:
            logger.info(f"  Probabilidad máxima seleccionada: {probas[mask].max():.6f}")
            logger.info(f"  Probabilidad mínima seleccionada: {probas[mask].min():.6f}")
        
        return clientes_selected, umbral_teorico, len(clientes_selected)
    
    def estimate_expected_gain(
        self,
        probas: np.ndarray,
        clientes_selected: np.ndarray,
        numero_de_cliente: np.ndarray,
        ganancia_acierto: int = 780000,
        costo_estimulo: int = 20000
    ) -> dict:
        """
        Estima ganancia esperada basada en probabilidades.
        
        Ganancia esperada = sum(proba_i * ganancia - costo) para clientes seleccionados
        """
        # Crear mask de seleccionados
        mask_selected = np.isin(numero_de_cliente, clientes_selected)
        
        # Probabilidades de los seleccionados
        probas_selected = probas[mask_selected]
        
        # Ganancia esperada por cliente
        ganancia_esperada_individual = probas_selected * ganancia_acierto - costo_estimulo
        
        # Ganancia total esperada
        ganancia_total_esperada = ganancia_esperada_individual.sum()
        
        # Estadísticas
        n_positivo_esperado = probas_selected.sum()  # Número esperado de churns detectados
        
        result = {
            'ganancia_total_esperada': float(ganancia_total_esperada),
            'n_envios': len(clientes_selected),
            'n_positivos_esperados': float(n_positivo_esperado),
            'ganancia_promedio_por_envio': float(ganancia_total_esperada / len(clientes_selected)) if len(clientes_selected) > 0 else 0,
            'probabilidad_media_seleccionados': float(probas_selected.mean()) if len(probas_selected) > 0 else 0
        }
        
        logger.info(f"\nGanancia esperada estimada:")
        logger.info(f"  Total esperado: ${result['ganancia_total_esperada']:,.0f}")
        logger.info(f"  Envíos: {result['n_envios']:,}")
        logger.info(f"  Churns esperados detectados: {result['n_positivos_esperados']:.1f}")
        logger.info(f"  Ganancia promedio por envío: ${result['ganancia_promedio_por_envio']:,.0f}")
        logger.info(f"  Probabilidad media: {result['probabilidad_media_seleccionados']:.4f}")
        
        return result
    
    def save_submission(
        self,
        clientes_selected: np.ndarray,
        filename: str = "submission.csv"
    ):
        """Guarda CSV para la competencia"""
        output_path = self.final_output_dir / filename
        
        df = pl.DataFrame({
            'numero_de_cliente': clientes_selected
        })
        
        # Ordenar por numero_de_cliente
        df = df.sort('numero_de_cliente')
        
        df.write_csv(output_path)
        
        logger.info(f"\n✓ Submission guardado: {output_path}")
        logger.info(f"  Total clientes: {len(df):,}")
        
        return output_path
    
    def run_prediction(self):  # ← INDENTADO CORRECTAMENTE
        """Pipeline completo de predicción para la competencia"""
        
        ganancia_acierto = self.ensemble_config['test']['GANANCIA_ACIERTO']
        costo_estimulo = self.ensemble_config['test']['COSTO_ESTIMULO']
        models_config = self.ensemble_config['models']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PREDICCIÓN FINAL PARA COMPETENCIA")
        logger.info(f"{'='*70}")
        
        # Agrupar modelos por study
        models_by_study = defaultdict(list)
        for model_file in self.models_dir.glob("*.pkl"):
            study_name = model_file.stem.rsplit('_seed', 1)[0]
            models_by_study[study_name].append(model_file)
        
        logger.info(f"\nModelos encontrados:")
        for study, files in models_by_study.items():
            logger.info(f"  {study}: {len(files)} modelos")
        
        # Predecir con cada study en su dataset
        # CAMBIO: Guardar en DataFrame de Polars para alinear
        study_dataframes = []
        
        for model_info in models_config:
            study_name = model_info['study_name']
            
            if study_name not in models_by_study:
                logger.warning(f"⚠️  No se encontraron modelos para {study_name}, saltando...")
                continue
            
            logger.info(f"\n{'='*70}")
            logger.info(f"PREDICIENDO: {study_name}")
            logger.info(f"{'='*70}")
            
            # Cargar config del modelo
            model_config = self.load_model_config(model_info['config_dir'])
            
            # Determinar data_path
            if 'DATA_PATH_OPT' in model_config:
                data_path = model_config['DATA_PATH_OPT']
            else:
                raise ValueError(f"No se encontró DATA_PATH en config de {study_name}")
            
            # Crear tabla en DuckDB
            table_name = f"final_{study_name}_202108"
            data_path_expanded = os.path.expanduser(data_path)
            
            conn = create_sql_table_from_parquet(data_path_expanded, table_name)
            
            try:
                # Predecir
                study_proba, clientes = self.predict_by_study(
                    study_name,
                    models_by_study[study_name],
                    conn,
                    table_name
                )
                
                # Guardar en DataFrame
                df_study = pl.DataFrame({
                    'numero_de_cliente': clientes,
                    f'proba_{study_name}': study_proba
                })
                
                study_dataframes.append(df_study)
                
            finally:
                conn.close()
                gc.collect()
        
        # CAMBIO: Hacer JOIN de todos los DataFrames por numero_de_cliente
        logger.info(f"\n{'='*70}")
        logger.info(f"ALINEANDO {len(study_dataframes)} MODELOS POR CLIENTE")
        logger.info(f"{'='*70}")
        
        # Join iterativo
        df_ensemble = study_dataframes[0]
        for df_study in study_dataframes[1:]:
            df_ensemble = df_ensemble.join(
                df_study, 
                on='numero_de_cliente', 
                how='inner'  # INNER JOIN: solo clientes que están en TODOS los datasets
            )
        
        logger.info(f"Clientes comunes a todos los modelos: {len(df_ensemble):,}")
        
        # Verificar que no perdimos muchos clientes
        for i, df_study in enumerate(study_dataframes):
            n_original = len(df_study)
            n_final = len(df_ensemble)
            pct_perdido = (n_original - n_final) / n_original * 100
            logger.info(f"  Dataset {i+1}: {n_original:,} → {n_final:,} ({pct_perdido:.2f}% perdido)")
        
        # Calcular promedio de probabilidades
        proba_cols = [col for col in df_ensemble.columns if col.startswith('proba_')]
        
        df_ensemble = df_ensemble.with_columns([
            pl.mean_horizontal(proba_cols).alias('probabilidad_ensemble')
        ])
        
        ensemble_proba = df_ensemble['probabilidad_ensemble'].to_numpy()
        numero_de_cliente = df_ensemble['numero_de_cliente'].to_numpy()
        
        logger.info(f"\nEstadísticas del ensemble:")
        logger.info(f"  Total clientes: {len(numero_de_cliente):,}")
        logger.info(f"  Probabilidad media: {ensemble_proba.mean():.4f}")
        logger.info(f"  Probabilidad máxima: {ensemble_proba.max():.4f}")
        logger.info(f"  Probabilidad mínima: {ensemble_proba.min():.4f}")
        
        # Guardar probabilidades completas
        probas_file = self.final_output_dir / "probabilidades_202108.parquet"
        df_ensemble.write_parquet(probas_file)
        logger.info(f"\n✓ Probabilidades guardadas: {probas_file}")
        
        # Estrategia 1: Top 11,000
        logger.info(f"\n{'='*70}")
        logger.info(f"ESTRATEGIA 1: TOP 11,000 CLIENTES")
        logger.info(f"{'='*70}")
        
        clientes_top11k, umbral_top11k = self.select_clients_fixed_threshold(
            ensemble_proba, numero_de_cliente, n_envios=11000
        )
        
        gain_est_top11k = self.estimate_expected_gain(
            ensemble_proba, clientes_top11k, numero_de_cliente,
            ganancia_acierto, costo_estimulo
        )
        
        submission_top11k = self.save_submission(
            clientes_top11k, "submission_top11000.csv"
        )
        
        # Estrategia 2: Umbral de probabilidad
        logger.info(f"\n{'='*70}")
        logger.info(f"ESTRATEGIA 2: UMBRAL DE PROBABILIDAD")
        logger.info(f"{'='*70}")
        
        clientes_umbral, umbral_prob, n_envios_umbral = self.select_clients_probability_threshold(
            ensemble_proba, numero_de_cliente, ganancia_acierto, costo_estimulo
        )
        
        gain_est_umbral = self.estimate_expected_gain(
            ensemble_proba, clientes_umbral, numero_de_cliente,
            ganancia_acierto, costo_estimulo
        )
        
        submission_umbral = self.save_submission(
            clientes_umbral, "submission_probability_threshold.csv"
        )
        
        # Comparación
        logger.info(f"\n{'='*70}")
        logger.info(f"COMPARACIÓN DE ESTRATEGIAS")
        logger.info(f"{'='*70}")
        
        logger.info(f"\nEstrategia 1 (Top 11,000):")
        logger.info(f"  Ganancia esperada: ${gain_est_top11k['ganancia_total_esperada']:,.0f}")
        logger.info(f"  Envíos: {gain_est_top11k['n_envios']:,}")
        
        logger.info(f"\nEstrategia 2 (Umbral probabilidad):")
        logger.info(f"  Ganancia esperada: ${gain_est_umbral['ganancia_total_esperada']:,.0f}")
        logger.info(f"  Envíos: {gain_est_umbral['n_envios']:,}")
        
        # Mejor estrategia
        if gain_est_top11k['ganancia_total_esperada'] > gain_est_umbral['ganancia_total_esperada']:
            mejor = "Top 11,000"
            archivo_mejor = submission_top11k
            ganancia_mejor = gain_est_top11k['ganancia_total_esperada']
        else:
            mejor = "Umbral probabilidad"
            archivo_mejor = submission_umbral
            ganancia_mejor = gain_est_umbral['ganancia_total_esperada']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"RECOMENDACIÓN")
        logger.info(f"{'='*70}")
        logger.info(f"Mejor estrategia: {mejor}")
        logger.info(f"Ganancia esperada: ${ganancia_mejor:,.0f}")
        logger.info(f"Archivo: {archivo_mejor}")
        
        # Guardar resumen
        import json
        summary = {
            'fecha_prediccion': str(pl.datetime('now')),
            'n_studies': len(study_dataframes),
            'studies': proba_cols,
            'total_clientes_202108': int(len(numero_de_cliente)),
            'estrategia_1_top11000': {
                'archivo': str(submission_top11k),
                **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                   for k, v in gain_est_top11k.items()}
            },
            'estrategia_2_umbral': {
                'archivo': str(submission_umbral),
                'umbral_probabilidad': float(umbral_prob),
                **{k: float(v) if isinstance(v, (int, float, np.number)) else v 
                   for k, v in gain_est_umbral.items()}
            },
            'recomendacion': {
                'estrategia': mejor,
                'archivo': str(archivo_mejor),
                'ganancia_esperada': float(ganancia_mejor)
            }
        }
        
        summary_file = self.final_output_dir / "prediction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n✓ Resumen guardado: {summary_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python predict_final_competition.py <ensemble_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    predictor = CompetitionPredictor(config_path)
    predictor.run_prediction()