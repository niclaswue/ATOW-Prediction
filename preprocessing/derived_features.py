from preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
from utils.dataset import Dataset


class DerivedFeaturePreprocessor(BasePreprocessor):

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding derived features")
        # add day of week
        dataset.df["day"] = pd.to_datetime(dataset.df.date).dt.day
        dataset.df["month"] = pd.to_datetime(dataset.df.date).dt.month
        dataset.df["year"] = pd.to_datetime(dataset.df.date).dt.year
        dataset.df["day_of_week"] = pd.to_datetime(dataset.df.date).dt.day_of_week
        dataset.df["quarter"] = pd.to_datetime(dataset.df.date).dt.quarter
        dataset.df["is_week_day"] = dataset.df["day_of_week"] < 5
        print("Done.")
        return dataset
