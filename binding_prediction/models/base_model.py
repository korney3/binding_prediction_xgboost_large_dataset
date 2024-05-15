from abc import ABC

from binding_prediction.config.config import Config


class Model(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def train(self, train_dataset, val_dataset):
        pass

    def save(self, path):
        pass

    def predict(self, data):
        pass

    def load(self, path):
        pass