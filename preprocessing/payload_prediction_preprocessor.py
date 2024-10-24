from utils.dataset import Dataset
from tqdm import tqdm
import numpy as np
from preprocessing.base_preprocessor import BasePreprocessor

from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor


class PayloadPredictionPreprocessor(BasePreprocessor):
    def __init__(self, model_path, no_cache=False) -> None:
        super().__init__(no_cache)
        assert Path(model_path).exists()
        self.model = TabularPredictor.load(model_path)

    def process(self, dataset: Dataset) -> Dataset:
        data = dataset.df.copy()

        # TODO: Use IATA PLF data
        # https://www.iata.org/en/iata-repository/publications/economic-reports/air-passenger-market-analysis---december-2022/
        average_plf = 0.812

        seat_plf = {  # just some guesses
            "Seats First_Class": 0.50,
            "Seats First_Class_Suite": 0.50,
            "Seats Business_Class": 0.70,
            "Seats Economy_Comfort_Class": 0.80,
            "Seats Economy_Family_Couch": 0.80,
            "Seats Premium_Economy_Class": 0.80,
            "Seats Economy_Class": 0.92,
        }

        total_seats = data["Seats Total"]

        # TODO: around 10 flights have no entries about seat distrution
        seat_weighted_load_factor = np.clip(
            sum(data[st] * seat_plf[st] for st in seat_plf) / total_seats,
            0.6,
            1.0,
        )

        col_map = {
            "aircraft_type": "aircraft_type",
            "Seats Total": "SEATS",
            "air_time_hours": "AIR_TIME",
            "ramp_to_ramp_hours": "RAMP_TO_RAMP",
            "route_distance_mi": "DISTANCE",
            "month": "MONTH",
        }
        drop_cols = [c for c in data.columns if c not in col_map.keys()]
        data.drop(columns=drop_cols, inplace=True)
        data.rename(col_map, inplace=True, axis=1)

        data["AIR_TIME"] *= 60
        data["RAMP_TO_RAMP"] *= 60

        load_factors = {
            "seat_weighted_plf": seat_weighted_load_factor,
            # "fixed_plf_75": 0.75,
            # "fixed_plf_80": 0.8,
            # "fixed_plf_85": 0.85,
            # "fixed_plf_90": 0.9,
            # "fixed_plf_95": 0.95,
            # "fixed_plf_100": 1.0,
            "iata_plf_81": average_plf,
        }

        print("Predicting Payloads based on T100 filings...")
        for plf_name, series in tqdm(load_factors.items()):
            data["LOAD_FACTOR"] = series
            ag_dataset = TabularDataset(data)
            predictions = self.model.predict(ag_dataset)
            predictions_kg = predictions * 0.453592  # libs to kg
            dataset.df[f"predicted_kg_payload_{plf_name}"] = predictions_kg
            dataset.df[f"predicted_tow_t100_{plf_name}"] = (
                dataset.df["ZFW"] + predictions_kg
            )
        print("Done")

        return dataset
