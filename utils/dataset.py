import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, name=""):
        self.df = dataframe
        self.name = name

    def split(self, train_percent: float = 0.8, seed: int = 0):
        assert train_percent <= 1.0
        train = self.df.sample(frac=train_percent, random_state=seed)
        test = self.df.drop(train.index)
        return train, test

    def split(self, train_percent: float = 0.8, seed: int = 0):
        assert 0 < train_percent <= 1.0
        n = len(self.df)

        # Create a boolean mask for the train set
        mask = np.random.RandomState(seed).choice(
            [True, False], size=n, p=[train_percent, 1 - train_percent]
        )

        # Use the mask to split the dataframe
        train = self.df[mask]
        test = self.df[~mask]

        return train, test
