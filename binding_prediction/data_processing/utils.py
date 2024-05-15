from logging import Logger
import typing as tp
from binding_prediction.const import FeaturizerTypes
from binding_prediction.data_processing.base_featurizer import Featurizer


def get_featurizer(config, file_path, logger: tp.Optional[Logger] = None) -> Featurizer:
    if config.yaml_config.featurizer_config.name == FeaturizerTypes.CIRCULAR:
        from binding_prediction.data_processing.circular_fingerprint import CircularFingerprintFeaturizer
        featurizer = CircularFingerprintFeaturizer(config, file_path, logger=logger)
    elif config.yaml_config.featurizer_config.name == FeaturizerTypes.MACCS:
        from binding_prediction.data_processing.maccs_fingerprint import MACCSFingerprintFeaturizer
        featurizer = MACCSFingerprintFeaturizer(config, file_path, logger=logger)
    elif config.yaml_config.featurizer_config.name == FeaturizerTypes.ENSEMBLE_PREDICTIONS:
        from binding_prediction.data_processing.ensemble_predictions_fingerprint import EnsemblePredictionsFeaturizer
        featurizer = EnsemblePredictionsFeaturizer(config, file_path, logger=logger)
    else:
        raise NotImplementedError(f"Fingerprint "
                                  f"{config.yaml_config.featurizer_config.name} "
                                  f"is not implemented")
    return featurizer
