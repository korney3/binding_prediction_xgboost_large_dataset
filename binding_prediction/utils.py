import time
from dataclasses import dataclass


def calculate_number_of_neg_and_pos_samples(pq_file, pq_groups_numbers=None):
    neg_samples = 0
    pos_samples = 0
    if pq_groups_numbers is None:
        pq_groups_numbers = range(pq_file.metadata.num_row_groups)
    for group_id in pq_groups_numbers:
        group_df = pq_file.read_row_group(group_id).to_pandas()
        neg_samples += len(group_df[group_df['binds'] == 0])
        pos_samples += len(group_df[group_df['binds'] == 1])
    return neg_samples, pos_samples


@dataclass
class ModelTypes:
    XGBOOST = 'xgboost'
    XGBOOST_ENSEMBLE = 'xgboost_ensemble'


@dataclass
class FeaturizerTypes:
    CIRCULAR = 'circular_fingerprint'


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result

    return wrapper


def pretty_print_text(text):
    print(f"{'=' * len(text)}\n{text.upper()}\n{'=' * len(text)}")
