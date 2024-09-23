from utils.dataset import Dataset
import pandas as pd
from preprocessing.base_preprocessor import BasePreprocessor


class CleanDatasetPreprocessor(BasePreprocessor):
    def process(self, dataset: Dataset) -> Dataset:
        # TODO: Remove duplicates in data etc.
        datetime_cols = ["date", "actual_offblock_time", "arrival_time", "valid"]
        for c in datetime_cols:
            if c not in dataset.df.columns:
                continue
            dataset.df[c] = pd.to_datetime(dataset.df[c]).astype(int)

        # never seen in submission, only train data
        dataset.df = dataset.df[dataset.df["aircraft_type"] != "C56X"]
        return dataset
