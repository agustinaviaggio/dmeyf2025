# train_ensemble.py
import yaml
import optuna
import duckdb
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
from joblib import Parallel, delayed
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
    
    def get_all_training_periods(self, model_config: dict) -> List[int]:
        """
        Obtiene TODOS los períodos de entrenamiento: train + test1 + test2
        
        Parameters:
        -----------
        model_config : dict
            Configuración del modelo
        
        Returns:
        --------
        List[int]
            Lista de períodos ordenados para entrenamiento
        """
        # Detectar períodos de train según estrategia
        if 'PERIODOS_TRAIN' in model_config:
            periodos_train = model_config['PERIODOS_TRAIN']
        elif 'PERIODOS_CLASE_1' in model_config:
            periodos_train = model_config['PERIODOS_CLASE_1']
        else:
            raise ValueError("No se encontró configuración de períodos de entrenamiento")
        
        # Obtener períodos de test
        periodos_test = []
        if 'MES_TEST_1' in model_config:
            periodos_test.extend(model_config['MES_TEST_1'])
        if 'MES_TEST_2' in model_config:
            periodos_test.extend(model_config['MES_TEST_2'])
        
        # Combinar y ordenar
        all_periodos = sorted(set(periodos_train + periodos_test))
        
        logger.info(f"Períodos train originales: {periodos_train}")
        logger.info(f"Períodos test: {periodos_test}")
        logger.info(f"Períodos totales para entrenamiento: {all_periodos}")
        
        return all_periodos
    
    def prepare_undersampled_data(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        model_config: dict
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepara dataset con undersampling UNA vez con seed fija.
        Ahora incluye períodos de train + test1 + test2
        
        Returns:
            Tuple[X, y, feature_cols]
        """
        # Obtener TODOS los períodos (train + test)
        periodos = self.get_all_training_periods(model_config)
        
        # Detectar ratio de undersampling
        if 'UNDERSAMPLING_RATIO' in model_config:
            ratio = model_config['UNDERSAMPLING_RATIO']
        elif 'UNDERSAMPLING_RATIO_tscv' in model_config:
            ratio = model_config['UNDERSAMPLING_RATIO_tscv']
        else:
            raise ValueError("No se encontró configuración de UNDERSAMPLING_RATIO")
        
        periodos_str = ','.join(map(str, periodos))
        
        # Query SIMPLE: traer TODO
        query = f"SELECT * FROM {table_name} WHERE foto_mes IN ({periodos_str})"
        
        logger.info(f"Cargando datos para undersampling de {len(periodos)} períodos...")
        data = conn.execute(query).fetchnumpy()
        
        # Hacer undersampling EN PYTHON con seed fija
        mask_clase_1 = data['target_binario'] == 1
        mask_clase_0 = data['target_binario'] == 0
        
        idx_clase_1 = np.where(mask_clase_1)[0]
        idx_clase_0 = np.where(mask_clase_0)[0]
        
        # Sample exacto de clase 0 con SEED FIJA
        n_clase_1 = len(idx_clase_1)
        n_clase_0_sample = int(n_clase_1 / ratio)
        
        np.random.seed(600011)  # SEED FIJA para que todas las semillas usen el mismo undersampling
        idx_clase_0_sample = np.random.choice(idx_clase_0, size=n_clase_0_sample, replace=False)
        
        # Combinar indices
        idx_train = np.concatenate([idx_clase_1, idx_clase_0_sample])
        
        logger.info(f"Dataset undersampled: {len(idx_train):,} registros "
                   f"(Clase 0: {n_clase_0_sample:,}, Clase 1: {n_clase_1:,})")
        
        # Preparar features
        feature_cols = [
            col for col in data.keys() 
            if col not in ['target_binario', 'target_ternario', 'foto_mes']
        ]
        
        X = np.column_stack([data[col][idx_train] for col in feature_cols])
        y = data['target_binario'][idx_train]
        
        # Limpiar
        del data
        gc.collect()
        
        return X, y, feature_cols
    
    def train_single_seed(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str],
        best_params: dict,
        best_iteration: int,
        semilla: int,
        study_name: str
    ) -> dict:
        """
        Entrena un modelo con una semilla específica.
        Esta función se ejecuta en paralelo.
        """
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
            'n_jobs': 1,  # Importante: 1 thread por modelo en paralelo
            'seed': semilla,
            **best_params
        }
        
        # Entrenar
        train_set = lgb.Dataset(X, label=y, feature_name=feature_cols)
        
        model = lgb.train(
            params,
            train_set,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
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
        
        # Limpiar
        del train_set, model
        gc.collect()
        
        return {
            'semilla': semilla,
            'model_path': str(model_path),
            'success': True
        }
    
    def train_all(self):
        """Pipeline completo de entrenamiento de todos los modelos"""
        
        models_config = self.ensemble_config['models']
        all_results = []
        
        for model_info in models_config:
            study_name = model_info['study_name']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"MODELO: {study_name}")
            logger.info(f"{'='*70}")
            
            conn = None
            
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
                
                # 5. Preparar dataset con undersampling UNA vez (ahora incluye test)
                logger.info(f"Preparando dataset con undersampling (train + test)...")
                X, y, feature_cols = self.prepare_undersampled_data(
                    conn, table_name, model_config
                )
                
                # 6. Entrenar TODAS las semillas en paralelo (batches de 6)
                logger.info(f"\nEntrenando {len(self.semillas)} modelos en paralelo (6 workers)...")
                
                start_time = datetime.now()
                
                results = Parallel(n_jobs=6, verbose=10)(
                    delayed(self.train_single_seed)(
                        X, y, feature_cols, best_params, best_iteration, semilla, study_name
                    )
                    for semilla in self.semillas
                )
                
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                
                logger.info(f"\n✓ {study_name} completado en {elapsed:.1f} segundos")
                logger.info(f"  Modelos entrenados: {len(results)}")
                
                all_results.extend(results)
                
                # Limpiar
                del X, y
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error en modelo {study_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                
            finally:
                if conn is not None:
                    try:
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
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
            
            logger.info(f"\n{'='*70}")
            logger.info(f"ENTRENAMIENTO COMPLETO")
            logger.info(f"Total modelos entrenados: {len(all_results)}")
            logger.info(f"Resultados: {results_csv}")
            logger.info(f"Modelos: {self.models_dir}")
            logger.info(f"{'='*70}")
            
            return results_df
        else:
            logger.error("No se entrenó ningún modelo exitosamente")
            return pl.DataFrame()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "ensemble_config.yaml"
    
    trainer = EnsembleTrainer(config_path)
    results = trainer.train_all()
    
    print("\n=== RESUMEN FINAL ===")
    if len(results) > 0:
        print(results.group_by('study_name').agg([
            pl.count('semilla').alias('n_modelos')
        ]))