import os
import time

import pandas as pd


def get_submission_test_predictions(test_dataset, test_Xy, model, logs_dir):
    start_time = time.time()
    test_pred = model.predict(test_Xy)
    submission = pd.DataFrame(columns=['id', 'binds'])
    print(f"Testing model time: {time.time() - start_time}")
    print('Saving predictions')
    start_time = time.time()
    for group_id in range(test_dataset.parquet_file.metadata.num_row_groups):
        group_df = test_dataset.parquet_file.read_row_group(group_id).to_pandas()
        submission = pd.concat([submission,
                                pd.DataFrame({
                                    'id': group_df['id'],
                                    'binds':
                                        test_pred[
                                        group_id * test_dataset.shard_size:
                                        (group_id + 1) * test_dataset.shard_size]})])
    submission.to_csv(os.path.join(logs_dir, 'submission.csv'), index=False)
    print(f"Saving predictions time: {time.time() - start_time}")
