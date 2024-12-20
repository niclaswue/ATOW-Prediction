import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils.dataset import Dataset
from functools import cache
from preprocessing.base_preprocessor import BasePreprocessor

root_dir = Path(__file__).parent.parent.absolute()

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


class WeatherDataPreprocessor(BasePreprocessor):

    @cache
    def weather_data(self):
        print("Loading weather data...")
        wdf = pd.read_csv(
            root_dir / "additional_data" / "weather_data" / "all_weather.tsv",
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

    def process(self, dataset: Dataset):
        weather = self.weather_data()
        dataset.df["arrival_time"] = pd.to_datetime(
            dataset.df["arrival_time"], utc=True
        )

        weather_dfs = []

        for airport in tqdm(dataset.df.ades.unique(), desc="Processing airports"):
            wdf = weather[weather.station == airport].sort_values(["valid"])
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
