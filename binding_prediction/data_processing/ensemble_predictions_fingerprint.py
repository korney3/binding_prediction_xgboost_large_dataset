import os.path
import time

import numpy as np

from binding_prediction.config.config import Config
from binding_prediction.const import WEAK_LEARNER_ARTIFACTS_NAME_PREFIX
from binding_prediction.data_processing.base_featurizer import Featurizer
from binding_prediction.evaluation.utils import get_predictions
from binding_prediction.utils import load_config


class EnsemblePredictionsFeaturizer(Featurizer):
    def __init__(self, config: Config,
                 pq_file_path: str):
        super().__init__(config, pq_file_path)
        self.config = config

    def featurize(self):
        start_time = time.time()
        self.x = self.smiles_to_predictions_fingerprint()
        print(f"Fingerprinting time: {time.time() - start_time}")
        start_time = time.time()
        self.create_target()
        print(f"Combining time: {time.time() - start_time}")

    def smiles_to_predictions_fingerprint(self):
        num_weak_learners = self.config.yaml_config.model_config.num_weak_learners
        parent_logs_dir = self.config.logs_dir
        features = np.zeros((len(self.indices_in_shard), num_weak_learners))
        for i in range(num_weak_learners):
            weak_learner_config = load_config(
                os.path.join(parent_logs_dir, f"{WEAK_LEARNER_ARTIFACTS_NAME_PREFIX}{i}"))

            predictions = get_predictions(weak_learner_config, self.pq_file_path, self.indices_in_shard)
            features[:, i] = predictions
        return features
