import time
import typing as tp
from functools import partial
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from binding_prediction.config.featurizer_config import CircularFingerprintFeaturizerConfig, \
    MACCSFingerprintFeaturizerConfig
from binding_prediction.const import TARGET_COLUMN
from binding_prediction.data_processing.base_featurizer import Featurizer


class MACCSFingerprintFeaturizer(Featurizer):
    def __init__(self, config: MACCSFingerprintFeaturizerConfig,
                 pq_file_path: str, protein_map: tp.Dict[str, int],
                 indices: tp.List[int] = None):
        super().__init__(config, pq_file_path, protein_map, indices)
        self.config = config

    def featurize(self):
        start_time = time.time()
        with Pool(8) as p:
            x = np.array(p.map(smiles_to_maccs_fingerprint,
                               self.smiles))
        print(f"Fingerprinting time: {time.time() - start_time}")
        start_time = time.time()
        self.x = np.array([x[i] + [self.proteins_encoded[i]] for i in range(len(x))])
        if TARGET_COLUMN in self.row_group_df.columns:
            self.y = np.array(self.row_group_df[TARGET_COLUMN])
        else:
            self.y = np.array([-1] * len(x))
        print(f"Combining time: {time.time() - start_time}")


def smiles_to_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr