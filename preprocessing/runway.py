import pandas as pd
from utils.dataset import Dataset
from tqdm import tqdm
from functools import cache

runway_info = pd.read_csv("runway_data/runways.csv")


@cache
def info_for_airport(airport):
    df = runway_info[runway_info["airport_ident"] == airport]
    df = df[df["closed"] == False]
    relevant_cols = [
        "length_ft",
        "he_elevation_ft",
        "le_elevation_ft",
        "he_displaced_threshold_ft",
        "le_displaced_threshold_ft",
    ]
    return df[relevant_cols].max()


def add_runway_data(dataset: Dataset):
    print("Adding runway data...")

    tqdm.pandas()

    # Apply info_for_airport to 'ades' and 'adep'
    ades_info = dataset.df["ades"].progress_apply(lambda x: info_for_airport(x))
    adep_info = dataset.df["adep"].progress_apply(lambda x: info_for_airport(x))

    # Add new columns for 'ades' information
    for col in ades_info.columns:
        dataset.df[f"runway_ades_{col}"] = ades_info[col]

    # Add new columns for 'adep' information
    for col in adep_info.columns:
        dataset.df[f"runway_adep_{col}"] = adep_info[col]

    print("Done.")
    return dataset
