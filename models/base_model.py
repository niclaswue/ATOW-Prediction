import pandas as pd
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abstractmethod
    def train(self, training_df: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, input_df: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def info(self):
        return {}
