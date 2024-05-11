# Prediction of Protein Binding - Kaggle Competition "Leash Bio - Predict New Medicines with BELKA"

Kaggle competition
page: [Leash Bio - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview)

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

Best test MAP score - 0.390

## Train xgboost model

```bash
python binding_prediction/xgboost_training_pipeline.py --input_parquet PATH_TO_INPUT_TRAIN_PARQUET_FILE \
                                                       --test_parquet PATH_TO_INPUT_TEST_PARQUET_FILE \
                                                       --config_path PATH_TO_YAML_WITH_CONFIG
```

If You want to run it in debug mode, add `--debug` flag.
