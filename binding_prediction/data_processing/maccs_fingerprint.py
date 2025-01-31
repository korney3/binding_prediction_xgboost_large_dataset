import typing as tp
from logging import Logger

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from binding_prediction.config.config import Config
from binding_prediction.data_processing.base_featurizer import Featurizer


class MACCSFingerprintFeaturizer(Featurizer):
    def __init__(self, config: Config,
                 pq_file_path: str, logger: tp.Optional[Logger] = None):
        super().__init__(config, pq_file_path, logger=logger)

    def featurize(self):
        self._featurize(smiles_to_maccs_fingerprint)


def smiles_to_maccs_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMACCSKeysFingerprint(mol)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr