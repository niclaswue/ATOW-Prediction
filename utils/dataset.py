import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


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

    def k_fold_split(self, k: int = 5, seed: int = 0):
        """
        Perform k-fold cross-validation splitting on the dataset.

        Args:
            k (int): Number of folds. Default is 5.
            seed (int): Random seed for reproducibility. Default is 0.

        Returns:
            list: A list of tuples, where each tuple contains (train_fold, test_fold)
                  for each split in the k-fold cross-validation.
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        splits = []

        for train_index, test_index in kf.split(self.df):
            train_fold = self.df.iloc[train_index]
            test_fold = self.df.iloc[test_index]
            splits.append((train_fold, test_fold))

        return splits
