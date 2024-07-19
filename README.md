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
So protein data is included into training features only as LabelEncoded feature (mapping can be seen [here](./data/processed/protein_map.json)). 
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


# Usage

## Train XGBoost model


### Config files

Currently available config files:
- `binding_prediction/config/yamls/xgboost_config.yaml` - config for training a single XGBoost model with circular fingerprints
- `binding_prediction/config/yamls/xgboost_maccs_config.yaml` - config for training a single XGBoost model with MACCS fingerprints

Configs are in YAML format and have the following structure:
```yaml
model: # Parameters related to XGBoost model architecture
  name: 'xgboost'
  eta: 0.05
  max_depth: 10
  objective: 'binary:logistic'
  eval_metric: 'map'
  verbosity: 2
  nthread: 50
  tree_method: "hist"
  grow_policy: 'depthwise'
  subsample: 0.6
  colsample_bytree: 0.6
  num_boost_round: 100
  alpha: 0.05
  device: 'cpu' # Set to 'gpu' if you have GPU 
train:
  early_stopping_rounds: 5
  train_size: 26000000 # Number of samples from input parquet file to use for training
featurizer:
  name: 'circular_fingerprint' # or 'maccs_fingerprint'
  radius: 3
  length: 1024
```
For more information about fields in config files, please refer to the [Config module](binding_prediction/config/config.py).

### Run training

```bash
python binding_prediction/run_training.py --input_parquet PATH_TO_INPUT_TRAIN_PARQUET_FILE \
                                          --test_parquet PATH_TO_INPUT_TEST_PARQUET_FILE \
                                          --config_path PATH_TO_YAML_WITH_CONFIG
```

If You want to run it in debug mode, add `--debug` flag. It will run training on a small subset of data.

### Output

After training is finished, you'll see new directory with timestamp of training start in the `logs` directory.
Inside this directory, you'll find:
- `model.pkl` - trained XGBoost model
- `model_{boosting_round}.pkl` - trained XGBoost model for each boosting round
- `config.yaml` - config file used for training
- `train_indices.npy` - indices of samples in input parquet file used for training
- `val_indices.npy` - indices of samples in input parquet file used for testing
- `{input_parquet_file_name}_metrics.csv` - Average Precision, ROC AUC and Accuracy metrics for validation set for final model
- `submission.csv` - submission file with predictions for test set in the format required by Kaggle competition


## Train XGBoost Ensemble model

### Config files

Currently available config files:
- `binding_prediction/config/yamls/xgboost_ensemble_config.yaml` - config for training ensemble of XGBoost models


Configs are in YAML format and have the following structure:
```yaml
model:
  name: 'xgboost_ensemble'
  weak_learner_config: # Config for each weak learner. Here you can use any config from xgboost_config.yaml
    model:
      name: 'xgboost'
      eta: 0.05
      max_depth: 10
      objective: 'binary:logistic'
      eval_metric: 'map'
      verbosity: 2
      nthread: 50
      tree_method: "hist"
      grow_policy: 'depthwise'
      subsample: 0.6
      colsample_bytree: 0.6
      num_boost_round: 100
      alpha: 0.05
      device: 'cpu' # Set to 'gpu' if you have GPU
    train:
      early_stopping_rounds: 5
      train_size: 26000000
    featurizer:
      name: 'circular_fingerprint'
      radius: 3
      length: 1024
  eta: 0.05 # This is start of model parameters for ensemble model itself
  max_depth: 10
  objective: 'binary:logistic'
  eval_metric: 'map'
  verbosity: 2
  nthread: 50
  tree_method: "hist"
  grow_policy: 'depthwise'
  subsample: 0.6
  colsample_bytree: 0.6
  num_boost_round: 100
  alpha: 0.05
  device: 'cpu' # Set to 'gpu' if you have GPU
train:
  early_stopping_rounds: 5
  train_size: 26000000
featurizer:
  name: 'ensemble_predictions'
```

```bash
python binding_prediction/run_training.py --input_parquet PATH_TO_INPUT_TRAIN_PARQUET_FILE \
                                          --test_parquet PATH_TO_INPUT_TEST_PARQUET_FILE \
                                          --config_path binding_prediction/config/yamls/xgboost_ensemble_config.yaml
```

If You want to run it in debug mode, add `--debug` flag. It will run training on a small subset of data.

### Output

After training is finished, you'll see new directory with timestamp of training start in the `logs` directory.
Inside this directory, you'll find all the same files as for single XGBoost model training, which will refer to the ensemble model.
Also for each weak learner, they will be stored in subdirectories of log directory with names `weak_learner_{i}`.

