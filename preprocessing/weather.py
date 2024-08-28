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
    )
    wdf["valid"] = pd.to_datetime(wdf.valid, utc=True)
    wdf = wdf.sort_values(by="valid")
    wdf = wdf.replace("M", 0)
    return wdf


@cache
def weather_change_features(ades, arr_time):
    wdf = weather_data()

    # Filter the dataframe for the specific airport
    closest = wdf[(wdf.station == ades) & (wdf.valid == arr_time)]

    # one_h_before = df[(df.valid > arr_time - timedelta(hours=1))].head(1)
    # one_h_after = df[(df.valid < arr_time + timedelta(hours=1))].tail(1)
    # three_h_before = df[(df.valid > arr_time - timedelta(hours=3))].head(1)
    # three_h_after = df[(df.valid < arr_time + timedelta(hours=3))].tail(1)

    # result = pd.concat(
    #     [closest, one_h_before, one_h_after, three_h_before, three_h_after]
    # )[cols]

    return closest  # result


def add_weather_data(dataset: Dataset):
    tqdm.pandas()
    dataset.df["arrival_time"] = pd.to_datetime(dataset.df["arrival_time"])
    dataset.df["rounded_arrival_time"] = dataset.df["arrival_time"].dt.round("30min")
    metar = dataset.df.progress_apply(
        lambda x: weather_change_features(x.ades, x.rounded_arrival_time), axis=1
    )
    from IPython import embed

    embed()
    exit()  # TODO: Remove DBG

    # dataset.df["mtow"] = dataset.df["aircraft_type"].progress_apply(
    #     lambda x: props_for_aircraft(x)["mtow"]
    # )
    # dataset.df["mlw"] = dataset.df["aircraft_type"].progress_apply(
    #     lambda x: props_for_aircraft(x)["mlw"]
    # )
    return dataset
