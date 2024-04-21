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
python binding_prediction/preproces_and_save_featurized_data.py --input_parquet data/train.parquet --train_set 1
# Test set
python binding_prediction/preproces_and_save_featurized_data.py --input_parquet data/test.parquet --train_set 0
```