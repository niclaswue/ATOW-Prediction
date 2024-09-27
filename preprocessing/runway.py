import pandas as pd
from utils.dataset import Dataset
from tqdm import tqdm
from functools import cache
from pathlib import Path
from preprocessing.base_preprocessor import BasePreprocessor

root_dir = Path(__file__).parent.parent


class RunwayInfoPreprocessor(BasePreprocessor):
    @cache
    def info_for_airport(self, airport):
        file = root_dir / "additional_data" / "runway_data" / "runways.csv"
        runway_info = pd.read_csv(file)
        df = runway_info[runway_info["airport_ident"] == airport]
        df = df[df["closed"] is False]
        relevant_cols = [
            "length_ft",
            "he_elevation_ft",
            "le_elevation_ft",
            "he_displaced_threshold_ft",
            "le_displaced_threshold_ft",
        ]
        return df[relevant_cols].max()  # we only return the max for all values

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding runway data...")

        tqdm.pandas()

        # Apply info_for_airport to 'ades' and 'adep'
        ades_info = dataset.df["ades"].progress_apply(
            lambda x: self.info_for_airport(x)
        )
        adep_info = dataset.df["adep"].progress_apply(
            lambda x: self.info_for_airport(x)
        )

        # Add new columns for 'ades' information
        for col in ades_info.columns:
            dataset.df[f"runway_ades_{col}"] = ades_info[col]

        # Add new columns for 'adep' information
        for col in adep_info.columns:
            dataset.df[f"runway_adep_{col}"] = adep_info[col]

        print("Done.")
        return dataset
