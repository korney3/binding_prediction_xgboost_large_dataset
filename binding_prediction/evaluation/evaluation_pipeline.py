import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost
from sklearn.metrics import roc_auc_score, average_precision_score

from binding_prediction.config.config import Config
from binding_prediction.const import ModelTypes
from binding_prediction.const import TARGET_COLUMN
from binding_prediction.data_processing.utils import get_featurizer
from binding_prediction.datasets.xgboost_iterator import SmilesIterator
from binding_prediction.evaluation.kaggle_submission_creation import get_submission_test_predictions_for_xgboost_model
from binding_prediction.models.xgboost_model import XGBoostModel
from binding_prediction.utils import timing_decorator, pretty_print_text, get_indices_in_shard


class EvaluationPipeline:
    def __init__(self, config: Config,
                 prediction_pq_file_path: str,
                 debug: bool = False,
                 rng: np.random.Generator = np.random.default_rng(seed=42),
                 prediction_indices=None):
        self.config = config

        self.debug = debug
        self.rng = rng

        self.model = self.load_model()

        self.prediction_pq_file_path = prediction_pq_file_path

        self.prediction_pq = pq.ParquetFile(self.prediction_pq_file_path)
        if prediction_indices is not None:
            self.prediction_indices = prediction_indices
        else:
            self.prediction_indices = np.arange(self.prediction_pq.metadata.num_rows)

    def run(self):
        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):
            dataset, matrix_Xy = self.prepare_data()
            predictions = self.model.predict(matrix_Xy)
            return predictions
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")

    def load_model(self):
        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):
            model = XGBoostModel(self.config)
            model.load(os.path.join(self.config.logs_dir, 'model.pkl'))
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")
        return model

    def calculate_metrics(self, predictions):
        pretty_print_text("Calculating metrics")
        shard_size = self.prediction_pq.metadata.row_group(0).num_rows
        targets = []
        for group_id in range(self.prediction_pq.metadata.num_row_groups):
            group_df = self.prediction_pq.read_row_group(group_id).to_pandas()
            if group_df[TARGET_COLUMN].isna().sum() == group_df.shape[0]:
                break
            _, relative_indices = get_indices_in_shard(self.prediction_indices, group_id, shard_size)
            targets.extend(group_df[TARGET_COLUMN].values)
        if len(targets) == 0:
            print("File for predictions does not contain any targets")
            return None, None, None

        roc_auc = roc_auc_score(targets, predictions)
        average_precision = average_precision_score(targets, predictions)
        accuracy = np.mean((predictions > 0.5) == targets)
        prediction_file_name = os.path.basename(self.prediction_pq_file_path).split('.')[0]
        print(f"Metrics for {prediction_file_name}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Average precision: {average_precision}")
        print(f"Accuracy: {accuracy}")
        metrics = pd.DataFrame({
            'roc_auc': [roc_auc],
            'average_precision': [average_precision],
            'accuracy': [accuracy]
        })

        metrics.to_csv(os.path.join(self.config.logs_dir, f'{prediction_file_name}_metrics.csv'), index=False)
        return roc_auc, average_precision, accuracy

    @timing_decorator
    def create_kaggle_submission_file(self):
        pretty_print_text("Testing model")
        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):
            test_dataset, test_Xy = self.prepare_data()
            get_submission_test_predictions_for_xgboost_model(test_dataset, test_Xy,
                                                              self.model, self.config.logs_dir)
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")

    def prepare_data(self,):
        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):
            featurizer = get_featurizer(self.config, self.prediction_pq_file_path)
            dataset = SmilesIterator(self.config, featurizer, self.prediction_pq_file_path,
                                     indicies=self.prediction_indices,
                                     shuffle=False)
            dmatrix_Xy = xgboost.DMatrix(dataset)
            return dataset, dmatrix_Xy
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")

