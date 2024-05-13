import os.path
import time
import typing as tp

import numpy as np
import pandas as pd

from binding_prediction.config.config import Config
from binding_prediction.data_processing.base_featurizer import Featurizer


class EnsemblePredictionsFeaturizer(Featurizer):
    def __init__(self, config: Config,
                 pq_file_path: str, protein_map: tp.Dict[str, int],
                 relative_indices: tp.List[int] = None,
                 indices_in_shard: tp.List[int] = None):
        super().__init__(config, pq_file_path, protein_map, relative_indices)
        self.config = config
        self.indices_in_shard = indices_in_shard

    def featurize(self):
        start_time = time.time()
        self.x = self.smiles_to_predictions_fingerprint()
        print(f"Fingerprinting time: {time.time() - start_time}")
        start_time = time.time()
        self.create_target()
        print(f"Combining time: {time.time() - start_time}")

    def smiles_to_predictions_fingerprint(self):
        num_weak_learners = self.config.yaml_config.model_config.num_weak_learners
        logs_dir = self.config.logs_dir
        features = np.zeros((len(self.indices_in_shard), num_weak_learners))
        for i in range(num_weak_learners):
            prediction = pd.read_csv(os.path.join(logs_dir, f"final_ensemble_model_predictions_{i}.csv"))
            prediction.set_index('index_in_train_file', inplace=True)
            features[:, i] = prediction.loc[self.indices_in_shard, 'prediction'].values
        return features
