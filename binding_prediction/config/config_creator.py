import typing as tp
from functools import partial

from binding_prediction.config.featurizer_config import load_circular_fingerprint_featurizer_config_from_yaml_path, \
    create_circular_fingerprint_featurizer_config_from_dict, load_maccs_fingerprint_featurizer_config_from_yaml_path, \
    create_maccs_fingerprint_featurizer_config_from_dict, load_ensemble_predictions_featurizer_config_from_yaml_path, \
    create_ensemble_predictions_featurizer_config_from_dict
from binding_prediction.config.model_config import load_xgboost_model_config_from_yaml_path, \
    create_xgboost_model_config_from_dict, load_xgboost_ensemble_model_config_from_yaml_path, \
    create_xgboost_ensemble_model_config_from_dict
from binding_prediction.config.training_config import load_training_config_from_yaml_path, \
    create_training_config_from_dict
from binding_prediction.const import ModelTypes, FeaturizerTypes


class ConfigCreator:
    def __init__(self, config: tp.Union[str, dict],
                 function_create_from_yaml_path: tp.Callable[[str], tp.Any],
                 function_create_from_dict: tp.Callable[[dict], tp.Any]):
        self.config = config
        self.function_create_from_yaml_path = function_create_from_yaml_path
        self.function_create_from_dict = function_create_from_dict

    def create(self):
        if isinstance(self.config, str):
            return self.function_create_from_yaml_path(self.config)
        elif isinstance(self.config, dict):
            return self.function_create_from_dict(self.config)
        else:
            raise ValueError(f"Config type {type(self.config)} is not supported")


def get_model_config(config: tp.Union[str, dict], model_name, neg_samples: int, pos_samples: int):
    if model_name == ModelTypes.XGBOOST:
        scale_pos_weight = neg_samples / pos_samples
        yaml_func = partial(load_xgboost_model_config_from_yaml_path, scale_pos_weight=scale_pos_weight)
        dict_func = partial(create_xgboost_model_config_from_dict, scale_pos_weight=scale_pos_weight)
        model_config_creator = ConfigCreator(config, yaml_func, dict_func)
    elif model_name == ModelTypes.XGBOOST_ENSEMBLE:
        scale_pos_weight = neg_samples / pos_samples
        yaml_func = partial(load_xgboost_ensemble_model_config_from_yaml_path, scale_pos_weight=scale_pos_weight)
        dict_func = partial(create_xgboost_ensemble_model_config_from_dict, scale_pos_weight=scale_pos_weight)
        model_config_creator = ConfigCreator(config, yaml_func, dict_func)
    else:
        raise ValueError(f"Model {model_name} is not supported")
    config = model_config_creator.create()
    return config


def get_featurizer_config(config: tp.Union[str, dict], featurizer_name):
    if featurizer_name == FeaturizerTypes.CIRCULAR:
        featurizer_config_creator = ConfigCreator(config,
                                                  load_circular_fingerprint_featurizer_config_from_yaml_path,
                                                  create_circular_fingerprint_featurizer_config_from_dict)
    elif featurizer_name == FeaturizerTypes.MACCS:
        featurizer_config_creator = ConfigCreator(config,
                                                  load_maccs_fingerprint_featurizer_config_from_yaml_path,
                                                  create_maccs_fingerprint_featurizer_config_from_dict)
    elif featurizer_name == FeaturizerTypes.ENSEMBLE_PREDICTIONS:
        featurizer_config_creator = ConfigCreator(config, load_ensemble_predictions_featurizer_config_from_yaml_path,
                                                  create_ensemble_predictions_featurizer_config_from_dict)
    else:
        raise ValueError(f"Featurizer {featurizer_name} is not supported")
    featurizer_config = featurizer_config_creator.create()
    return featurizer_config


def get_training_config(config: tp.Union[str, dict]):
    training_config_creator = ConfigCreator(config, load_training_config_from_yaml_path,
                                            create_training_config_from_dict)
    config = training_config_creator.create()
    return config


