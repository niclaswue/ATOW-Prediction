from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset
import numpy as np


class SampleWeightPreprocessor(BasePreprocessor):
    def __init__(
        self,
        no_cache=True,
        max_weight_ratio: float = 3.0,
    ):
        super().__init__(no_cache)
        self.max_weight_ratio = max_weight_ratio

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding sample weights")

        distances = dataset.df["flown_distance"]
        min_dist = distances.min()
        max_dist = distances.max()

        # Linear scaling between 1 and max_weight_ratio
        normalized_distances = (distances - min_dist) / (max_dist - min_dist)
        dataset.df["sample_weight"] = 1 + normalized_distances * (
            self.max_weight_ratio - 1
        )

        # Normalize weights to sum to number of samples
        n_samples = len(dataset.df)
        dataset.df["sample_weight"] = dataset.df["sample_weight"] * (
            n_samples / dataset.df["sample_weight"].sum()
        )

        print("Done.")
        return dataset
