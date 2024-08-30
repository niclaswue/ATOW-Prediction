import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils.dataset import Dataset
from functools import cache
from datetime import datetime, timedelta
import time

pd.set_option("future.no_silent_downcasting", True)

cols = [
    "valid",
    "tmpf",
    "dwpf",
    "relh",
    "drct",
    "sknt",
    "p01i",
    "alti",
    "mslp",
    "vsby",
    "gust",
    "skyc1",
    "skyc2",
    "skyc3",
    "skyc4",
    "skyl1",
    "skyl2",
    "skyl3",
    "skyl4",
    "ice_accretion_1hr",
    "ice_accretion_3hr",
    "ice_accretion_6hr",
    "peak_wind_gust",
    "peak_wind_drct",
    "peak_wind_time",
    "feel",
    "snowdepth",
]


@cache
def weather_data():
    print("Loading weather data...")
    wdf = pd.read_csv(
        Path(__file__).parents[1] / "weather_data" / "all_weather.tsv",
        sep="\t",
        index_col=0,
        low_memory=False,
        dtype={col: str for col in cols if col.startswith("skyc")},
    )
    wdf["valid"] = pd.to_datetime(wdf.valid, utc=True)

    # Convert numeric columns
    numeric_cols = [
        "tmpf",
        "dwpf",
        "relh",
        "drct",
        "sknt",
        "p01i",
        "alti",
        "mslp",
        "vsby",
        "gust",
        "skyl1",
        "skyl2",
        "skyl3",
        "skyl4",
        "ice_accretion_1hr",
        "ice_accretion_3hr",
        "ice_accretion_6hr",
        "peak_wind_gust",
        "peak_wind_drct",
        "feel",
        "snowdepth",
    ]
    for col in numeric_cols:
        wdf[col] = pd.to_numeric(wdf[col].replace("M", "0"), errors="coerce")

    print("Done")
    return wdf


@cache
def weather_for_airport(airport):
    wdf = weather_data()
    return wdf[wdf.station == airport].sort_values(["valid"])


def add_weather_data(dataset: Dataset):
    dataset.df["arrival_time"] = pd.to_datetime(dataset.df["arrival_time"])

    weather_dfs = []

    for airport in tqdm(dataset.df.ades.unique(), desc="Processing airports"):
        wdf = weather_for_airport(airport)
        # rename cols with prefix
        wdf = wdf.add_prefix("ades_")
        wdf = wdf.rename(columns={"ades_valid": "valid", "ades_station": "ades"})

        mask = dataset.df.ades == airport
        adf = dataset.df[mask].sort_values(["arrival_time"])

        merged = pd.merge_asof(
            left=adf,
            right=wdf,
            left_on="arrival_time",
            right_on="valid",
            by="ades",
            direction="nearest",
            tolerance=pd.Timedelta("1h"),
        )

        merged = merged.drop(columns=["ades_metar"])

        weather_dfs.append(merged)

    dataset.df = pd.concat(weather_dfs, ignore_index=True)

    # Sort the DataFrame back to its original order
    dataset.df = dataset.df.sort_index()

    return dataset


# Example usage
if __name__ == "__main__":
    # Assuming you have a Dataset object
    dataset = Dataset(...)  # Initialize your dataset
    dataset_with_weather = add_weather_data(dataset)

    # Further processing or analysis can be done here
    print(dataset_with_weather.df.head())
