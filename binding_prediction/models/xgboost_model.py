import pickle

import xgboost


class XGBoostModel:
    def __init__(self, params: dict, num_rounds: int,
                 early_stopping_rounds: int = 5):
        self.params = params
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None

    def train(self, train_Xy, eval_list):
        self.model = xgboost.train(self.params, train_Xy, self.num_rounds,
                                   evals=eval_list, verbose_eval=True,
                                   early_stopping_rounds=self.early_stopping_rounds)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, data):
        return self.model.predict(data)

    def load(self, path):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
