# Prediction of Protein Binding - Kaggle Competition "Leash Bio - Predict New Medicines with BELKA"

Kaggle competition page: [Leash Bio - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview)

# Installation
```bash
conda env create -f environment.yml
conda activate binding_prediction
pip install -e .
```

# Download data
Get the data from the [Kaggle competition page](https://www.kaggle.com/competitions/leash-BELKA/data) and save it in the `data` folder.

# Featurize data
```bash
# Train set
python binding_prediction/preproces_and_save_circular_featurized_data.py --input_parquet data/train.parquet
# Test set
python binding_prediction/preproces_and_save_circular_featurized_data.py --input_parquet data/test.parquet
```

# XGBoost model baseline


## Run model in debug regime
```bash
python binding_prediction/xgboost_training_pipeline.py --input_parquet data/row_group_0.parquet --debug
```

## Train xgboost baseline model
```bash
python binding_prediction/xgboost_training_pipeline.py
```