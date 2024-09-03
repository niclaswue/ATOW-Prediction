from utils.dataset import Dataset
import pandas as pd


# TODO: Remove duplicates in data etc.
def clean_dataset(dataset: Dataset):
    datetime_cols = ["date", "actual_offblock_time", "arrival_time", "valid"]
    for c in datetime_cols:
        if c not in dataset.df.columns:
            continue
        dataset.df[c] = pd.to_datetime(dataset.df[c]).astype(int)
    return dataset
