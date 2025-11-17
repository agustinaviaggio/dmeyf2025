import yaml
import optuna
import duckdb
import numpy as np
import polars as pl
from pathlib import Path
import tempfile
import sys
from typing import Dict, List, Tuple
import joblib
from tqdm import tqdm
import logging
import os
from datetime import datetime
import gc
import lightgbm as lgb
from src.features import create_sql_table_from_parquet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class EnsembleTrainer:
    def __init__(self, ensemble_config_path: str):
        """
        Inicializa el trainer de ensemble.
        
        Args:
            ensemble_config_path: Ruta al YAML de configuración del ensemble
        """
        with open(ensemble_config_path) as f:
            self.ensemble_config = yaml.safe_load(f)
        
        self.output_dir = Path(self.ensemble_config['ensemble']['output_dir']).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorio para modelos entrenados
        self.models_dir = self.output_dir / "trained_models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Directorio para predicciones en test
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
        
        # Cargar semillas
        self.semillas = self.ensemble_config['ensemble']['semillas']
        
        logger.info(f"Ensemble output directory: {self.output_dir}")
        logger.info(f"Semillas a usar: {len(self.semillas)}")
    
    def download_study_db(self, db_path: str) -> str:
        """Expande path del .db"""
        db_path_expanded = os.path.expanduser(db_path)
        
        if not Path(db_path_expanded).exists():
            raise FileNotFoundError(f"No se encontró DB en {db_path_expanded}")
        
        logger.info(f"Usando DB: {db_path_expanded}")
        return db_path_expanded
    
    def get_best_params(self, study_db_path: str, study_name: str) -> Tuple[dict, int]:
        """
        Extrae mejores hiperparámetros y best_iteration del study.
        
        Returns:
            Tuple[dict, int]: (mejores_params, best_iteration)
        
        Raises:
            ValueError: Si no se encuentra best_iteration en user_attrs
        """
        storage_url = f"sqlite:///{study_db_path}"
        storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": 30}}
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        best_params = study.best_params
        
        # Verificar que exista best_iteration
        if 'best_iteration' not in study.best_trial.user_attrs:
            error_msg = (f"No se encontró 'best_iteration' en user_attrs del study '{study_name}'. "
                        f"El estudio no está completo o no guardó best_iteration correctamente.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        best_iteration = study.best_trial.user_attrs['best_iteration']
        
        logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Best iteration: {best_iteration}")
        logger.info(f"Mejores params: {best_params}")
        
        return best_params, best_iteration
    
    def load_model_config(self, config_dir: str) -> dict:
        """Carga el conf.yaml del modelo"""
        config_path = Path(config_dir).expanduser() / "conf.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"No se encontró {config_path}")
        
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
        
        # Extraer configuración
        model_config = {
            'STUDY_NAME': full_config.get('STUDY_NAME'),
            **full_config.get('configuracion', {})
        }
        
        return model_config
    
    def get_train_query(self, model_config: dict, semilla: int, 
                       table_name: str) -> str:
        """
        Genera query SQL para datos de entrenamiento según estrategia.
        """
        # Estrategia TSCV (1311_7, 1411_2, 1411_3)
        if 'PERIODOS_TRAIN' in model_config:
            periodos = model_config['PERIODOS_TRAIN']
            periodos_str = ','.join(map(str, periodos))
            
            # Buscar ratio (puede estar como UNDERSAMPLING_RATIO o UNDERSAMPLING_RATIO_tscv)
            ratio = model_config.get('UNDERSAMPLING_RATIO') or model_config.get('UNDERSAMPLING_RATIO_tscv')
            
            query = f"""
                WITH clase_0_sample AS (
                    SELECT * FROM {table_name}
                    WHERE foto_mes IN ({periodos_str}) 
                      AND target_binario = 0
                    USING SAMPLE {ratio * 100} PERCENT (bernoulli, {semilla})
                ),
                clase_1_completa AS (
                    SELECT * FROM {table_name}
                    WHERE foto_mes IN ({periodos_str}) 
                      AND target_binario = 1
                )
                SELECT * FROM clase_0_sample
                UNION ALL
                SELECT * FROM clase_1_completa
            """
            
            logger.info(f"Estrategia TSCV: {len(periodos)} períodos, undersampling={ratio}")
            
        # Estrategia por períodos separados (1511_1, 1511_2)
        elif 'PERIODOS_CLASE_1' in model_config and 'PERIODOS_CLASE_0' in model_config:
            periodos_1 = model_config['PERIODOS_CLASE_1']
            periodos_0 = model_config['PERIODOS_CLASE_0']
            ratio = model_config['UNDERSAMPLING_RATIO']
            
            periodos_1_str = ','.join(map(str, periodos_1))
            periodos_0_str = ','.join(map(str, periodos_0))
            
            query = f"""
                WITH clase_0_sample AS (
                    SELECT * FROM {table_name}
                    WHERE foto_mes IN ({periodos_0_str}) 
                      AND target_binario = 0
                    USING SAMPLE {ratio * 100} PERCENT (bernoulli, {semilla})
                ),
                clase_1_completa AS (
                    SELECT * FROM {table_name}
                    WHERE foto_mes IN ({periodos_1_str}) 
                      AND target_binario = 1
                )
                SELECT * FROM clase_0_sample
                UNION ALL
                SELECT * FROM clase_1_completa
            """
            
            logger.info(f"Estrategia períodos separados: "
                       f"clase_1={len(periodos_1)} períodos, "
                       f"clase_0={len(periodos_0)} períodos, "
                       f"ratio={ratio}")
        
        else:
            raise ValueError(
                f"Estrategia no reconocida. Config debe tener "
                f"PERIODOS_TRAIN o (PERIODOS_CLASE_1 + PERIODOS_CLASE_0)"
            )
        
        return query
    
    def train_single_model(
        self,
        study_name: str,
        best_params: dict,
        best_iteration: int,
        semilla: int,
        model_config: dict,
        conn: duckdb.DuckDBPyConnection,
        table_name: str
    ) -> dict:
        """
        Entrena un modelo con una semilla específica.
        
        Returns:
            dict con 'model', 'feature_cols', y metadata
        """
        # Obtener query de datos
        query_train = self.get_train_query(model_config, semilla, table_name)
        
        # Ejecutar query y obtener datos
        train_data = conn.execute(query_train).fetchnumpy()
        
        n_clase_0 = (train_data['target_binario'] == 0).sum()
        n_clase_1 = (train_data['target_binario'] == 1).sum()
        
        logger.info(f"  Datos: {len(train_data['target_binario']):,} registros "
                   f"(Clase 0: {n_clase_0:,}, Clase 1: {n_clase_1:,})")
        
        # Preparar features
        feature_cols = [
            col for col in train_data.keys() 
            if col not in ['target_binario', 'target_ternario', 'foto_mes']
        ]
        
        X_train = np.column_stack([train_data[col] for col in feature_cols])
        y_train = train_data['target_binario']
        
        # Configurar parámetros
        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'first_metric_only': True,
            'boost_from_average': True,
            'feature_pre_filter': False,
            'max_bin': 31,
            'verbose': -1,
            'is_unbalance': True,
            'bagging_freq': 1,
            'n_jobs': -1,
            'seed': semilla,
            **best_params
        }
        
        # Entrenar
        train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        # Limpiar
        del X_train, y_train, train_data, train_set
        gc.collect()
        
        result = {
            'model': model,
            'feature_cols': feature_cols,
            'n_train': n_clase_0 + n_clase_1,
            'n_clase_0': int(n_clase_0),
            'n_clase_1': int(n_clase_1)
        }
        
        return result
    
    def predict_on_test(
        self,
        model,
        feature_cols: List[str],
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        test_periods: List[str],
        ganancia_acierto: int,
        costo_estimulo: int
    ) -> dict:
        """
        Hace predicción en conjunto de test y calcula ganancia.
        
        Returns:
            dict con predicciones y métricas
        """
        # Query de test
        periodos_test_str = ','.join(map(str, test_periods))
        query_test = f"SELECT * FROM {table_name} WHERE foto_mes IN ({periodos_test_str})"
        
        test_data = conn.execute(query_test).fetchnumpy()
        
        # Preparar features (asegurar mismo orden)
        X_test = np.column_stack([test_data[col] for col in feature_cols])
        y_test = test_data['target_ternario']
        
        # Predecir
        y_pred_proba = model.predict(X_test)
        
        # Calcular ganancia óptima
        df_eval = pl.DataFrame({
            'y_true': y_test,
            'y_pred_proba': y_pred_proba
        })
        
        df_ordenado = df_eval.sort('y_pred_proba', descending=True)
        df_ordenado = df_ordenado.with_columns([
            pl.when(pl.col('y_true') == 1)
              .then(ganancia_acierto)
              .otherwise(-costo_estimulo)
              .cast(pl.Int64)
              .alias('ganancia_individual')
        ])
        df_ordenado = df_ordenado.with_columns([
            pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
        ])
        
        idx_max = df_ordenado.select(pl.col('ganancia_acumulada').arg_max()).item()
        ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
        
        n_envios = idx_max + 1
        
        result = {
            'y_pred_proba': y_pred_proba,
            'y_true': y_test,
            'ganancia': float(ganancia_maxima),
            'n_envios': int(n_envios),
            'n_test': len(y_test)
        }
        
        # Limpiar
        del test_data, X_test
        gc.collect()
        
        return result
    
    def train_all(self):
        """Pipeline completo de entrenamiento de todos los modelos"""
        
        models_config = self.ensemble_config['models']
        test_periods_1 = self.ensemble_config['test']['MES_TEST_1']
        test_periods_2 = self.ensemble_config['test']['MES_TEST_2']
        ganancia_acierto = self.ensemble_config['test']['GANANCIA_ACIERTO']
        costo_estimulo = self.ensemble_config['test']['COSTO_ESTIMULO']
        
        all_results = []
        all_predictions_test1 = []
        all_predictions_test2 = []
    
        for model_info in models_config:
            study_name = model_info['study_name']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"MODELO: {study_name}")
            logger.info(f"{'='*70}")
            
            conn = None  # Inicializar conn
            
            try:
                # 1. Cargar config del modelo
                model_config = self.load_model_config(model_info['config_dir'])
                
                # 2. Determinar data_path
                if 'DATA_PATH_OPT' in model_config:
                    data_path = model_config['DATA_PATH_OPT']
                else:
                    raise ValueError(f"No se encontró DATA_PATH en config de {study_name}")
                
                # 3. Crear tabla en DuckDB
                table_name = f"tabla_{study_name}"
                data_path_expanded = os.path.expanduser(data_path)
                
                conn = create_sql_table_from_parquet(data_path_expanded, table_name)
                
                # 4. Obtener mejores hiperparámetros
                logger.info(f"Cargando study DB...")
                local_db = self.download_study_db(model_info['study_db'])
                
                logger.info(f"Extrayendo mejores hiperparámetros...")
                best_params, best_iteration = self.get_best_params(local_db, study_name)
                
                # 5. Entrenar para cada semilla
                logger.info(f"\nEntrenando {len(self.semillas)} modelos...")
                
                for i, semilla in enumerate(tqdm(self.semillas, desc=f"{study_name}")):
                    try:
                        logger.info(f"\n--- Semilla {i+1}/{len(self.semillas)}: {semilla} ---")
                        
                        # Entrenar modelo
                        result = self.train_single_model(
                            study_name,
                            best_params,
                            best_iteration,
                            semilla,
                            model_config,
                            conn,
                            table_name
                        )
                        
                        model = result['model']
                        feature_cols = result['feature_cols']
                        
                        # Guardar modelo
                        model_filename = f"{study_name}_seed{semilla}.pkl"
                        model_path = self.models_dir / model_filename
                        
                        model_package = {
                            'model': model,
                            'feature_cols': feature_cols,
                            'best_params': best_params,
                            'best_iteration': best_iteration,
                            'semilla': semilla,
                            'study_name': study_name
                        }
                        
                        joblib.dump(model_package, model_path)
                        logger.info(f"  Modelo guardado: {model_filename}")
                        
                        # Predecir en TEST 1
                        logger.info(f"  Prediciendo en TEST 1 ({test_periods_1})...")
                        pred_test1 = self.predict_on_test(
                            model, feature_cols, conn, table_name,
                            test_periods_1, ganancia_acierto, costo_estimulo
                        )
                        
                        # Predecir en TEST 2
                        logger.info(f"  Prediciendo en TEST 2 ({test_periods_2})...")
                        pred_test2 = self.predict_on_test(
                            model, feature_cols, conn, table_name,
                            test_periods_2, ganancia_acierto, costo_estimulo
                        )
                        
                        logger.info(f"  TEST 1 - Ganancia: {pred_test1['ganancia']:,.0f}, "
                                f"Envíos: {pred_test1['n_envios']:,}")
                        logger.info(f"  TEST 2 - Ganancia: {pred_test2['ganancia']:,.0f}, "
                                f"Envíos: {pred_test2['n_envios']:,}")
                        
                        # Guardar predicciones
                        all_predictions_test1.append({
                            'study_name': study_name,
                            'semilla': semilla,
                            'y_pred_proba': pred_test1['y_pred_proba'],
                            'y_true': pred_test1['y_true']
                        })
                        
                        all_predictions_test2.append({
                            'study_name': study_name,
                            'semilla': semilla,
                            'y_pred_proba': pred_test2['y_pred_proba'],
                            'y_true': pred_test2['y_true']
                        })
                        
                        # Registrar resultado
                        result_record = {
                            'study_name': study_name,
                            'semilla': semilla,
                            'model_path': str(model_path),
                            'best_iteration': best_iteration,
                            'n_train': result['n_train'],
                            'n_clase_0': result['n_clase_0'],
                            'n_clase_1': result['n_clase_1'],
                            'ganancia_test1': pred_test1['ganancia'],
                            'n_envios_test1': pred_test1['n_envios'],
                            'ganancia_test2': pred_test2['ganancia'],
                            'n_envios_test2': pred_test2['n_envios']
                        }
                        
                        all_results.append(result_record)
                        
                    except Exception as e:
                        logger.error(f"Error en {study_name} - semilla {semilla}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            except Exception as e:
                logger.error(f"Error en modelo {study_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
            finally:
                # Cerrar conexión de este modelo si existe
                if conn is not None:
                    try:
                        conn.execute(f"DROP TABLE IF EXISTS tabla_{study_name}")
                        conn.close()
                        logger.info(f"Conexión cerrada para {study_name}")
                    except Exception as e:
                        logger.warning(f"Error al cerrar conexión de {study_name}: {e}")
                gc.collect()
        
        # Guardar resumen
        if len(all_results) > 0:
            results_df = pl.DataFrame(all_results)
            results_csv = self.output_dir / "training_summary.csv"
            results_df.write_csv(results_csv)
            
            # Guardar predicciones
            self._save_predictions(all_predictions_test1, 'test1')
            self._save_predictions(all_predictions_test2, 'test2')
            
            # Analizar ensembles
            self._analyze_ensembles(results_df, ganancia_acierto, costo_estimulo)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"ENTRENAMIENTO COMPLETO")
            logger.info(f"Total modelos entrenados: {len(all_results)}")
            logger.info(f"Resultados: {results_csv}")
            logger.info(f"Modelos: {self.models_dir}")
            logger.info(f"Predicciones: {self.predictions_dir}")
            logger.info(f"{'='*70}")
            
            return results_df
        else:
            logger.error("No se entrenó ningún modelo exitosamente")
            return pl.DataFrame()
    
    def _save_predictions(self, predictions: List[dict], test_name: str):
        """Guarda predicciones en formato eficiente"""
        # Guardar como parquet para cada combinación study_name + semilla
        for pred in predictions:
            filename = f"{pred['study_name']}_seed{pred['semilla']}_{test_name}.parquet"
            filepath = self.predictions_dir / filename
            
            df = pl.DataFrame({
                'y_pred_proba': pred['y_pred_proba'],
                'y_true': pred['y_true']
            })
            
            df.write_parquet(filepath)
        
        logger.info(f"Predicciones de {test_name} guardadas: {len(predictions)} archivos")
    
    def _analyze_ensembles(self, results_df: pl.DataFrame, 
                          ganancia_acierto: int, costo_estimulo: int):
        """
        Analiza dos estrategias de ensemble:
        1. Promedio simple de todas las 125 predicciones
        2. Promedio por modelo (25 predicciones) y luego entre modelos (5 promedios)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"ANÁLISIS DE ENSEMBLES")
        logger.info(f"{'='*70}")
        
        for test_name in ['test1', 'test2']:
            logger.info(f"\n--- {test_name.upper()} ---")
            
            # Cargar todas las predicciones de este test
            pred_files = list(self.predictions_dir.glob(f"*_{test_name}.parquet"))
            
            if len(pred_files) == 0:
                logger.warning(f"No se encontraron predicciones para {test_name}")
                continue
            
            # Leer todas las predicciones
            all_preds = []
            study_preds = {}  # Para agrupar por study
            
            for pred_file in pred_files:
                df_pred = pl.read_parquet(pred_file)
                y_true = df_pred['y_true'].to_numpy()
                y_pred = df_pred['y_pred_proba'].to_numpy()
                
                all_preds.append(y_pred)
                
                # Extraer study_name del filename
                study_name = pred_file.stem.split('_seed')[0]
                if study_name not in study_preds:
                    study_preds[study_name] = []
                study_preds[study_name].append(y_pred)
            
            # Estrategia 1: Promedio simple de todo
            y_pred_ensemble_simple = np.mean(all_preds, axis=0)
            ganancia_simple = self._calculate_ganancia(
                y_true, y_pred_ensemble_simple, ganancia_acierto, costo_estimulo
            )
            
            logger.info(f"\nEstrategia 1 - Promedio simple ({len(all_preds)} predicciones):")
            logger.info(f"  Ganancia: {ganancia_simple['ganancia']:,.0f}")
            logger.info(f"  Envíos: {ganancia_simple['n_envios']:,}")
            
            # Estrategia 2: Promedio por modelo, luego entre modelos
            study_averages = []
            for study_name, preds in study_preds.items():
                study_avg = np.mean(preds, axis=0)
                study_averages.append(study_avg)
                
                # Ganancia de cada modelo promediado
                ganancia_study = self._calculate_ganancia(
                    y_true, study_avg, ganancia_acierto, costo_estimulo
                )
                logger.info(f"\n  {study_name} (promedio {len(preds)} semillas):")
                logger.info(f"    Ganancia: {ganancia_study['ganancia']:,.0f}")
                logger.info(f"    Envíos: {ganancia_study['n_envios']:,}")
            
            # Promedio final entre modelos
            y_pred_ensemble_2stage = np.mean(study_averages, axis=0)
            ganancia_2stage = self._calculate_ganancia(
                y_true, y_pred_ensemble_2stage, ganancia_acierto, costo_estimulo
            )
            
            logger.info(f"\nEstrategia 2 - Promedio 2 etapas ({len(study_averages)} modelos):")
            logger.info(f"  Ganancia: {ganancia_2stage['ganancia']:,.0f}")
            logger.info(f"  Envíos: {ganancia_2stage['n_envios']:,}")
            
            # Comparación
            logger.info(f"\nComparación:")
            logger.info(f"  Diferencia: {ganancia_2stage['ganancia'] - ganancia_simple['ganancia']:,.0f}")
            
            # Guardar resultados del análisis
            analysis_results = {
                'test_name': test_name,
                'estrategia_1_ganancia': ganancia_simple['ganancia'],
                'estrategia_1_envios': ganancia_simple['n_envios'],
                'estrategia_2_ganancia': ganancia_2stage['ganancia'],
                'estrategia_2_envios': ganancia_2stage['n_envios'],
                'diferencia': ganancia_2stage['ganancia'] - ganancia_simple['ganancia']
            }
            
            # Guardar en JSON
            import json
            analysis_file = self.output_dir / f"ensemble_analysis_{test_name}.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
    
    def _calculate_ganancia(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                           ganancia_acierto: int, costo_estimulo: int) -> dict:
        """Calcula ganancia óptima dado y_true y probabilidades"""
        df_eval = pl.DataFrame({
            'y_true': y_true,
            'y_pred_proba': y_pred_proba
        })
        
        df_ordenado = df_eval.sort('y_pred_proba', descending=True)
        df_ordenado = df_ordenado.with_columns([
            pl.when(pl.col('y_true') == 1)
              .then(ganancia_acierto)
              .otherwise(-costo_estimulo)
              .cast(pl.Int64)
              .alias('ganancia_individual')
        ])
        df_ordenado = df_ordenado.with_columns([
            pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')
        ])
        
        idx_max = df_ordenado.select(pl.col('ganancia_acumulada').arg_max()).item()
        ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
        
        return {
            'ganancia': float(ganancia_maxima),
            'n_envios': int(idx_max + 1)
        }