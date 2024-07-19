from dataclasses import dataclass

TARGET_COLUMN = 'binds'
WHOLE_MOLECULE_COLUMN = 'molecule_smiles'
PROTEIN_COLUMN = 'protein_name'
PROTEIN_MAP_JSON_PATH = 'data/processed/protein_map.json'

WEAK_LEARNER_ARTIFACTS_NAME_PREFIX = 'weak_learner_'
FINAL_ENSEMBLE_MODEL_ARTIFACTS_NAME_PREFIX = 'final_ensemble_model_'
DEFAULT_NUM_WEAK_LEARNERS = 3


@dataclass
class ModelTypes:
    XGBOOST = 'xgboost'
    XGBOOST_ENSEMBLE = 'xgboost_ensemble'


@dataclass
class FeaturizerTypes:
    CIRCULAR = 'circular_fingerprint'
    MACCS = 'maccs_fingerprint'
    ENSEMBLE_PREDICTIONS = 'ensemble_predictions'