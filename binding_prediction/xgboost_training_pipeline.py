import argparse
import json
import os
import pickle
import time

import numpy as np
import pyarrow.parquet as pq
import xgboost

from binding_prediction.datasets.xgboost_iterator import SmilesIterator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, default='data/two_row_groups.parquet')
    parser.add_argument('--featurizer', type=str, default='circular')
    parser.add_argument('--circular_fingerprint_radius', type=int, default=3)
    parser.add_argument('--circular_fingerprint_length', type=int, default=2048)
    parser.add_argument('--debug', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
    logs_dir = os.path.join('logs', current_date)
    os.makedirs(logs_dir, exist_ok=True)
    print('Train validation split')
    start_time = time.time()
    train_file_path = args.input_parquet
    rng = np.random.default_rng(seed=42)
    train_val_pq = pq.ParquetFile(train_file_path)

    if args.debug:
        train_size = 1000
    else:
        train_size = train_val_pq.metadata.num_rows

    train_indices = rng.choice(train_size, int(0.8 * train_size), replace=False)
    val_indices = np.setdiff1d(np.arange(train_size), train_indices)
    print(f"Train validation split time: {time.time() - start_time}")

    print('Creating datasets')
    start_time = time.time()
    train_dataset = SmilesIterator(train_file_path, indicies=train_indices,
                                   fingerprint=args.featurizer,
                                   radius=args.circular_fingerprint_radius,
                                   nBits=args.circular_fingerprint_length,
                                   test_set=False)
    val_dataset = SmilesIterator(train_file_path, indicies=val_indices,
                                 fingerprint=args.featurizer,
                                 radius=args.circular_fingerprint_radius,
                                 nBits=args.circular_fingerprint_length, test_set=False)

    train_Xy = xgboost.DMatrix(train_dataset)
    val_Xy = xgboost.DMatrix(val_dataset)
    print(f"Creating datasets time: {time.time() - start_time}")
    print(f"Datasets sizes: train: {train_Xy.num_row()}, "
          f"validation: {val_Xy.num_row()}")
    print('Creating model')
    start_time = time.time()
    params = {
        'max_depth': 10,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'verbosity': 2,
        'nthread': 12
    }
    with open(os.path.join(logs_dir, 'params.json'), 'w') as file:
        json.dump(params, file)
    num_rounds = 10

    eval_list = [(train_Xy, 'train'), (val_Xy, 'eval')]
    print(f"Creating model time: {time.time() - start_time}")
    print('Training model')
    start_time = time.time()
    model = xgboost.train(params, train_Xy, num_rounds,
                          evals=eval_list, verbose_eval=True,
                          early_stopping_rounds=5)
    print(f"Training model time: {time.time() - start_time}")
    print('Saving model')
    start_time = time.time()
    with open(os.path.join(logs_dir, 'model.pkl'), 'wb') as file:
        pickle.dump(model, file)
    print(f"Saving model time: {time.time() - start_time}")


if __name__ == '__main__':
    main()
