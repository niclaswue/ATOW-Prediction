import pandas as pd


class Dataset:
    def __init__(self, dataframe: pd.DataFrame, name=""):
        self.df = dataframe
        self.name = name

    def split(self, train_percent: float = 0.8, seed: int = 0):
        assert train_percent <= 1.0
        train = self.df.sample(frac=train_percent, random_state=seed)
        test = self.df.drop(train.index)
        return train, test
