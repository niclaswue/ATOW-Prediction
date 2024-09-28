from preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
from utils.dataset import Dataset


class DerivedFeaturePreprocessor(BasePreprocessor):
    def process(self, dataset: Dataset) -> Dataset:
        print("Adding derived features")
        # add day of week
        dataset.df["date"] = pd.to_datetime(dataset.df["date"])
        dataset.df["actual_offblock_time"] = pd.to_datetime(
            dataset.df["actual_offblock_time"]
        )
        dataset.df["arrival_time"] = pd.to_datetime(dataset.df["arrival_time"])

        # 6min average taxi in time all airports 2022 (https://ansperformance.eu/economics/cba/standard-inputs/chapters/taxiing_times.html#:~:text=The%20taxi%2Dout%20time%20is,%2Dblock%20time%20(AIBT).)
        dataset.df["onblock_time"] = dataset.df["arrival_time"] + pd.Timedelta(
            6, unit="m"
        )
        dataset.df["ramp_to_ramp_time"] = (
            dataset.df["onblock_time"] - dataset.df["actual_offblock_time"]
        )

        dataset.df["day"] = dataset.df["date"].dt.day
        dataset.df["month"] = dataset.df["date"].dt.month
        dataset.df["year"] = dataset.df["date"].dt.year
        dataset.df["day_of_week"] = dataset.df["date"].dt.day_of_week
        dataset.df["week"] = dataset.df["date"].dt.isocalendar().week
        dataset.df["quarter"] = dataset.df["date"].dt.quarter
        dataset.df["is_week_day"] = dataset.df["day_of_week"] < 5

        dataset.df["route"] = dataset.df["adep"] + "_" + dataset.df["ades"]

        dataset.df["airline_aircraft"] = (
            dataset.df["airline"] + "_" + dataset.df["aircraft_type"]
        )

        print("Done.")
        return dataset
