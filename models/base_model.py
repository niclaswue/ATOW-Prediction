import pandas as pd
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def train(self, training_df: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, input_df: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def info(self):
        return {}
