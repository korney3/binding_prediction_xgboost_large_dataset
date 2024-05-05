import pickle

import xgboost

from binding_prediction.models.base_model import Model


class XGBoostEnsembleModel(Model):
    def __init__(self, config, num_pq_groups_per_model=4):
        super().__init__(config)
        self.num_pq_groups_per_model = num_pq_groups_per_model

    def train(self, train_indicies, val_indicies):
        params = self.config.model_config.__dict__
        num_rounds = params.pop('num_boost_round')
        early_stopping_rounds = self.config.training_config.early_stopping_rounds

        self.model = xgboost.train(params, train_Xy, num_rounds,
                                   evals=eval_list, verbose_eval=True,
                                   early_stopping_rounds=early_stopping_rounds)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, data):
        return self.model.predict(data)

    def load(self, path):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)