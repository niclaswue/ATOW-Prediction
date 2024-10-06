from preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
from utils.dataset import Dataset
import airportsdata
from timezonefinder import TimezoneFinder
from tqdm import tqdm
from functools import cache
from pathlib import Path

base_path = Path(__file__).parent.parent.absolute()


class AirportPreprocessor(BasePreprocessor):
    def __init__(self, no_cache=False) -> None:
        super().__init__(no_cache)
        self.our_airports = pd.read_csv(
            base_path / "additional_data/airport_data/airports.csv"
        )
        self.ap_data = airportsdata.load()
        self.tzf = TimezoneFinder()

    @cache
    def get_airport_data(self, code):
        airport = self.our_airports[self.our_airports["ident"] == code]
        if len(airport) > 0:
            airport = airport.iloc[0]
            lat, lon = (
                airport["latitude_deg"],
                airport["longitude_deg"],
            )
            timezone_str = self.tzf.timezone_at(lng=lon, lat=lat)

            return {
                "city": airport["municipality"],
                "region": airport["iso_region"],
                "continent": airport["continent"],
                "type": airport["type"],
                "elevation": airport["elevation_ft"],
                "lat": lat,
                "lon": lon,
                "tz": timezone_str,
            }
        elif code in self.ap_data:
            data = self.ap_data[code]
            return {
                "city": data["city"],
                "region": data["subd"],
                "continent": pd.NA,
                "type": pd.NA,
                "elevation": data["elevation"],
                "lat": data["lat"],
                "lon": data["lon"],
                "tz": data["tz"],
            }
        else:
            print(code)
            return {
                "city": pd.NA,
                "region": pd.NA,
                "continent": pd.NA,
                "type": pd.NA,
                "elevation": pd.NA,
                "lat": pd.NA,
                "lon": pd.NA,
                "tz": pd.NA,
            }

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding airport features")
        tqdm.pandas()

        cols = ["city", "region", "continent", "type", "elevation", "lat", "lon", "tz"]

        for col in tqdm(cols):
            dataset.df[f"adep_{col}"] = dataset.df["adep"].apply(
                lambda x: self.get_airport_data(x)[col]
            )
            dataset.df[f"ades_{col}"] = dataset.df["ades"].apply(
                lambda x: self.get_airport_data(x)[col]
            )

        def local_time(timezone, date):
            if pd.isna(timezone):
                return date
            return date.tz_convert(timezone)

        dataset.df["actual_offblock_time"] = pd.to_datetime(
            dataset.df["actual_offblock_time"]
        )
        dataset.df["arrival_time"] = pd.to_datetime(dataset.df["arrival_time"])

        dataset.df["adep_local_offblock_time"] = dataset.df[
            ["adep_tz", "actual_offblock_time"]
        ].progress_apply(
            lambda row: local_time(row["adep_tz"], row["actual_offblock_time"]), axis=1
        )

        dataset.df["ades_local_arrival_time"] = dataset.df[
            ["ades_tz", "arrival_time"]
        ].progress_apply(
            lambda row: local_time(row["ades_tz"], row["arrival_time"]), axis=1
        )
        print("Done.")
        return dataset
