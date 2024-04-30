import os
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
from pyarrow import parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from binding_prediction.const import PROTEIN_COLUMN, WHOLE_MOLECULE_COLUMN, TARGET_COLUMN


def smiles_to_fingerprint(smiles, nBits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def process_row_group(pq_file_path, row_group_number, output_dir, protein_map,
                      circular_fingerprint_radius, circular_fingerprint_length):
    if os.path.exists(os.path.join(output_dir, f'Commit_file_{row_group_number}.txt')):
        protein_map = np.load(os.path.join(output_dir, 'protein_map.npy'), allow_pickle=True).item()
        return protein_map
    input_smiles, x, y = create_circular_fingerprints_from_pq_row_group(pq_file_path, row_group_number, protein_map,
                                                                        circular_fingerprint_radius,
                                                                        circular_fingerprint_length)
    start_time = time.time()
    np.save(os.path.join(output_dir, f'x_{row_group_number}.npy'), x)
    np.save(os.path.join(output_dir, f'input_smiles_{row_group_number}.npy'), input_smiles)
    np.save(os.path.join(output_dir, f'y_{row_group_number}.npy'), y)
    with open(os.path.join(output_dir, f'Commit_file_{row_group_number}.txt'), 'w') as f:
        f.write('Commit')
    print(f"Saving time: {time.time() - start_time}")
    np.save(os.path.join(output_dir, 'protein_map.npy'), protein_map)
    return protein_map


def create_circular_fingerprints_from_pq_row_group(pq_file_path, row_group_number, protein_map,
                                                   circular_fingerprint_radius, circular_fingerprint_length):
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
    if TARGET_COLUMN in row_group_df.columns:
        y = np.array(row_group_df[TARGET_COLUMN])
    else:
        y = np.array([-1] * len(x))
    print(f"Combining time: {time.time() - start_time}")
    return input_smiles, x, y
