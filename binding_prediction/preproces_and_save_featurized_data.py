import argparse
import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from binding_prediction.data_processing.circular_fingerprints import smiles_to_fingerprint
from const import PROTEIN_COLUMN, WHOLE_MOLECULE_COLUMN, TARGET_COLUMN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, default='data/two_row_groups.parquet')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--train_set', type=int, default=1, choices=[0, 1])
    parser.add_argument('--featurizer', type=str, default='circular')
    parser.add_argument('--circular_fingerprint_radius', type=int, default=3)
    parser.add_argument('--circular_fingerprint_length', type=int, default=2048)
    return parser.parse_args()


def process_row_group(pq_file_path, row_group_number, output_dir, protein_map,
                      circular_fingerprint_radius, circular_fingerprint_length,
                      train_set = True):
    if os.path.exists(os.path.join(output_dir, f'Commit_file.txt')):
        return protein_map
    print(f"Processing row group {row_group_number}")
    start_time = time.time()
    row_group_df = pq.ParquetFile(pq_file_path).read_row_group(row_group_number).to_pandas()
    print(f"Reading time: {time.time() - start_time}")
    start_time = time.time()
    proteins_encoded = []
    for protein in row_group_df[PROTEIN_COLUMN]:
        if protein not in protein_map:
            protein_map[protein] = len(protein_map)
        proteins_encoded.append(protein_map[protein])
    print(f"Protein encoding time: {time.time() - start_time}")
    start_time = time.time()
    input_smiles = row_group_df[WHOLE_MOLECULE_COLUMN]
    partial_smiles_to_fingerprint = partial(smiles_to_fingerprint, nBits=circular_fingerprint_length,
                                            radius=circular_fingerprint_radius)
    with Pool(8) as p:
        x = np.array(p.map(partial_smiles_to_fingerprint, input_smiles))
    print(f"Fingerprinting time: {time.time() - start_time}")
    start_time = time.time()
    x = np.array([x[i] + [proteins_encoded[i]] for i in range(len(x))])
    if train_set:
        y = np.array(row_group_df[TARGET_COLUMN])
    else:
        y = np.array([-1] * len(x))
    print(f"Combining time: {time.time() - start_time}")
    start_time = time.time()
    np.save(os.path.join(output_dir, f'x_{row_group_number}.npy'), x)
    np.save(os.path.join(output_dir, f'input_smiles_{row_group_number}.npy'), input_smiles)
    np.save(os.path.join(output_dir, f'y_{row_group_number}.npy'), y)
    with open(os.path.join(output_dir, f'Commit_file.txt'), 'w') as f:
        f.write('Commit')
    print(f"Saving time: {time.time() - start_time}")

    return protein_map


def main():
    args = parse_args()
    parquet_file_path = args.input_parquet
    if args.featurizer == 'circular':
        subdirectory = f'circular_{args.circular_fingerprint_radius}_{args.circular_fingerprint_length}'
    else:
        raise ValueError(f"Featurizer {args.featurizer} not supported")
    if args.train_set:
        subdirectory = os.path.join(subdirectory, 'train')
        protein_map = {}
    else:
        protein_map = np.load(os.path.join(args.output_dir, subdirectory, 'train', 'protein_map.npy'),
                              allow_pickle=True).item()
        subdirectory = os.path.join(subdirectory, 'test')
    output_dir = os.path.join(args.output_dir, subdirectory)
    os.makedirs(output_dir, exist_ok=True)
    num_row_groups = pq.ParquetFile(parquet_file_path).metadata.num_row_groups
    for i in tqdm(range(num_row_groups)):
        print(f"Processing row group {i}")
        start_time = time.time()
        protein_map = process_row_group(parquet_file_path, i, output_dir, protein_map, args.circular_fingerprint_radius,
                                        args.circular_fingerprint_length, train_set = args.train_set)
        print(f"Total time: {time.time() - start_time}")

    np.save(os.path.join(output_dir, 'protein_map.npy'), protein_map)


if __name__ == '__main__':
    main()
