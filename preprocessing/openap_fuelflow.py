from openap import FuelFlow
from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset
from pathlib import Path
from tqdm import tqdm
from functools import cache
import pandas as pd


@cache
def ff_for_ac(ac):
    # as so little aircraft are supported, we choose a common baseline.
    fuelflow = FuelFlow(ac="A320")
    return fuelflow


class OpenAPFuelFlowPreprocessor(BasePreprocessor):
    def __init__(self, no_cache=False) -> None:
        super().__init__(no_cache)
        self.base_path = (
            Path(__file__).parent.parent / "additional_data" / "aircraft_data"
        )

    def process(self, dataset: Dataset) -> Dataset:
        result = []
        for _, row in tqdm(dataset.df.iterrows(), total=len(dataset.df)):
            fuelflow = ff_for_ac(row["aircraft_type"])
            alt = row["cruise_altitude"]
            tas = row["mean_cruise_speed"] + row["average_headwind"]
            open_ap_cruise_mass = row["openap_mlw"] + (
                (row["openap_mtow"] - row["openap_mlw"]) / 2
            )
            ff = fuelflow.enroute(mass=open_ap_cruise_mass, tas=tas, alt=alt)
            result.append(ff)

        dataset.df["cruise_fuel_flow_calculated"] = pd.Series(result)
        return dataset
