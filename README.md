# Prediction of Protein Binding - Kaggle Competition "Leash Bio - Predict New Medicines with BELKA"

This repo contains code for the prediction of protein binding between small molecules and proteins. 
The data comes from Kaggle competition: [Leash Bio - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview). 

The code allows to train XGBoost models with circular fingerprints and MACCS fingerprints and ensemble of XGBoost models.
Code is optimized for running training on large amount of data using [External Memory](https://xgboost.readthedocs.io/en/stable/python/examples/external_memory.html#sphx-glr-python-examples-external-memory-py) XGBoost regime.

I've been able to run the XGBoost model training on a **MacOS** machine with **32** GB of RAM and **16** CPU cores, 
fitting **26M** samples 
with **1024** CircularFingerprint features. For Linux machines with **32** GB of RAM, **8** CPU cores and **25** GB GPU 
I've been able to fit **50M** samples with **512** CircularFingerprint features.

### Current limitations:
- The code is optimized for running on a single machine with a large amount of RAM. It is not optimized for running on a cluster.
- Competition data is demonstrating binding between small molecules and only 3 proteins. 
So protein data is included into trainig features only as LabelEncoded feature. 
I think for datasets with more diverse protein data it worth to add meaningful embeddings encoding proteins

# Installation

```bash
conda env create -f environment.yml
conda activate binding_prediction
pip install -e .
```

# Download data

Get the data from the [Kaggle competition page](https://www.kaggle.com/competitions/leash-BELKA/data) and save it in
the `data` folder.

# XGBoost model

Best test MAP score - 0.414

## Train XGBoost model

Currently available config files:
- `binding_prediction/config/yamls/xgboost_config.yaml` - config for training a single XGBoost model with circular fingerprints
- `binding_prediction/config/yamls/xgboost_maccs_config.yaml` - config for training a single XGBoost model with MACCS fingerprints

```bash
python binding_prediction/xgboost_training_pipeline.py --input_parquet PATH_TO_INPUT_TRAIN_PARQUET_FILE \
                                                       --test_parquet PATH_TO_INPUT_TEST_PARQUET_FILE \
                                                       --config_path PATH_TO_YAML_WITH_CONFIG
```

If You want to run it in debug mode, add `--debug` flag.

## Train XGBoost Ensemble model

```bash
python binding_prediction/xgboost_ensemble_training_pipeline.py --input_parquet PATH_TO_INPUT_TRAIN_PARQUET_FILE \
                                                                --test_parquet PATH_TO_INPUT_TEST_PARQUET_FILE \
                                                                --config_path binding_prediction/config/yamls/xgboost_ensemble_config.yaml
```