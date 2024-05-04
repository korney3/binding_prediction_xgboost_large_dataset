import os

import numpy as np
import pyarrow.parquet as pq
import xgboost
import yaml

from binding_prediction.config.config import Config
from binding_prediction.datasets.xgboost_iterator import SmilesIterator
from binding_prediction.evaluation.kaggle_submission_creation import get_submission_test_predictions_for_xgboost_model
from binding_prediction.models.xgboost_model import XGBoostModel
from binding_prediction.utils import ModelTypes, timing_decorator, pretty_print_text


class TrainingPipeline:
    def __init__(self, config: Config,
                 debug: bool = False,
                 rng: np.random.Generator = np.random.default_rng(seed=42)):
        self.config = config
        self.debug = debug
        self.rng = rng

        self.model = None

    def run(self):
        if self.config.model_config.name == ModelTypes.XGBOOST:
            train_Xy, val_Xy = self.prepare_train_val_data()
            self.save_config()
            self.model = XGBoostModel(self.config)
            self.train(train_Xy, val_Xy)
            self.evaluate()
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    def save_config(self):
        with open(os.path.join(self.config.logs_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.config.__dict__, file)

    @timing_decorator
    def train(self, train_dataset, val_dataset):
        pretty_print_text("Training model")
        if self.config.model_config.name == ModelTypes.XGBOOST:
            eval_list = [(train_dataset, 'train'), (val_dataset, 'eval')]
            self.model.train(train_dataset, eval_list)
            self.model.save(os.path.join(self.config.logs_dir, 'model.pkl'))
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    @timing_decorator
    def evaluate(self):
        pretty_print_text("Testing model")
        if self.config.model_config.name == ModelTypes.XGBOOST:
            test_dataset, test_Xy = self.prepare_test_data()
            get_submission_test_predictions_for_xgboost_model(test_dataset, test_Xy,
                                                              self.model, self.config.logs_dir)
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    @timing_decorator
    def prepare_train_val_data(self):
        pretty_print_text("Preparing train and validation data")
        train_val_pq = pq.ParquetFile(self.config.train_file_path)
        if self.debug:
            train_size = 100000
        else:
            train_size = train_val_pq.metadata.num_rows

        train_indices, val_indices = self.get_train_val_indicies(train_val_pq, train_size)

        if self.config.model_config.name == ModelTypes.XGBOOST:
            train_dataset = SmilesIterator(self.config.train_file_path, indicies=train_indices,
                                           fingerprint=self.config.featurizer_config.name,
                                           radius=self.config.featurizer_config.radius,
                                           nBits=self.config.featurizer_config.length)

            self.config.protein_map_path = train_dataset.protein_map_path

            val_dataset = SmilesIterator(self.config.train_file_path, indicies=val_indices,
                                         fingerprint=self.config.featurizer_config.name,
                                         radius=self.config.featurizer_config.radius,
                                         nBits=self.config.featurizer_config.length,
                                         protein_map_path=self.config.protein_map_path)

            train_Xy = xgboost.DMatrix(train_dataset)
            val_Xy = xgboost.DMatrix(val_dataset)

            return train_Xy, val_Xy
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    def prepare_test_data(self):
        if self.config.model_config.name == ModelTypes.XGBOOST:

            test_dataset = SmilesIterator(self.config.test_file_path, shuffle=False,
                                          fingerprint=self.config.featurizer_config.name,
                                          radius=self.config.featurizer_config.radius,
                                          nBits=self.config.featurizer_config.length,
                                          protein_map_path=self.config.protein_map_path)
            test_Xy = xgboost.DMatrix(test_dataset)
            return test_dataset, test_Xy
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    def get_train_val_indicies(self, train_val_pq, train_size):
        train_val_indices = self.rng.choice(train_val_pq.metadata.num_rows,
                                            train_size,
                                            replace=False)
        train_indices = self.rng.choice(train_val_indices, int(0.5 * train_size), replace=False)
        val_indices = np.setdiff1d(train_val_indices, train_indices)
        return train_indices, val_indices
