from abc import ABC, abstractmethod
from joblib import Memory
from utils.dataset import Dataset

# max 2GB cache per Preprocessor
memory = Memory(".cache", bytes_limit="2G", verbose=1)


class BasePreprocessor(ABC):
    def __init__(self, no_cache=False) -> None:
        super().__init__()
        self.no_cache = no_cache

    @abstractmethod
    def process(self, dataset: Dataset) -> Dataset:
        """Should modify the dataset and return the updated version"""
        raise NotImplementedError()

    def apply(self, dataset: Dataset) -> Dataset:
        if self.no_cache:
            apply_func = self.process
        else:
            apply_func = memory.cache(self.process, ignore=["self"])
        return apply_func(dataset)
