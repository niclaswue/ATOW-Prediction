from pathlib import Path
from typing import List
from pprint import pprint
import warnings

import pandas as pd

from utils.data_loader import DataLoader

from preprocessing.base_preprocessor import BasePreprocessor
from preprocessing.clean_dataset import CleanDatasetPreprocessor
from preprocessing.aircraft_performance import AircraftPerformancePreprocessor
from preprocessing.fuel_price_preprocessor import FuelPricePreprocessor
from preprocessing.runway import RunwayInfoPreprocessor
from preprocessing.pax_flow_preprocessor import PaxFlowPreprocessor
from preprocessing.weather import WeatherDataPreprocessor
from preprocessing.derived_features import DerivedFeaturePreprocessor

from models.autogluon_model import AutogluonModel

from evals.metrics import MetricEvals


pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

PREPROCESSORS: List[BasePreprocessor] = [
    AircraftPerformancePreprocessor(),
    FuelPricePreprocessor(),
    RunwayInfoPreprocessor(),
    PaxFlowPreprocessor(),
    WeatherDataPreprocessor(),
    DerivedFeaturePreprocessor(),
    CleanDatasetPreprocessor(),
]

evaluator = MetricEvals()
model = AutogluonModel(time_limit=5 * 60)


def main():
    loader = DataLoader(Path("data"), num_days=1, seed=1337)
    challenge, submission, final_submission = loader.load()

    for preprocessor in PREPROCESSORS:
        challenge = preprocessor.apply(challenge)

    train_df, val_df = challenge.split(train_percent=0.8)

    print(f"\n\nTraining model {model.name}")
    model.train(train_df)
    info = model.info()

    print("-" * 10)
    print(f"Model info: {pprint(info)}")
    print("-" * 10)
    predictions = model.predict(val_df)

    # evaluate the predictions per model
    evaluation = evaluator.evaluate(val_df.tow, predictions)
    for k, v in evaluation.items():
        print(f"{k:<20}: {v:.3f}")
    print("-" * 10)


if __name__ == "__main__":
    main()
