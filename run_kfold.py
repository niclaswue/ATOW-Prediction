import os
from pathlib import Path
from typing import List
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
from preprocessing.airport_preprocessor import AirportPreprocessor

from models.autogluon_model import AutogluonModel
from evals.metrics import MetricEvals
import argparse
import wandb

pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--quality", type=str, help="Quality Preset", default="best_quality"
)
parser.add_argument("--time", type=int, help="Time Limit (s)", default=30)
args = parser.parse_args()

PREPROCESSORS: List[BasePreprocessor] = [
    AirportPreprocessor(),
    AircraftPerformancePreprocessor(),
    FuelPricePreprocessor(),
    RunwayInfoPreprocessor(),
    PaxFlowPreprocessor(),
    WeatherDataPreprocessor(),
    DerivedFeaturePreprocessor(),
    CleanDatasetPreprocessor(),
]

model_config = {
    "time_limit": args.time,
    "preset": args.quality,
    "verbosity": 2,
}

evaluator = MetricEvals()
loader = DataLoader(Path("data"), num_days=0)


def train(dataset):
    for preprocessor in PREPROCESSORS:
        dataset = preprocessor.apply(dataset)

    models = []
    evaluations = []
    for i, (train_df, val_df) in enumerate(dataset.k_fold_split()):
        model = AutogluonModel(**model_config)
        print(f"\n\nTraining model {model.name} on split {i+1}")
        model.train(train_df)
        predictions = model.predict(val_df)
        evaluation = evaluator.evaluate(val_df.tow, predictions)
        evaluations.append(evaluation)
        model.log_feature_importance(train_df)
        models.append(models)
    return models, evaluations


if __name__ == "__main__":
    wandb.init(project="flying_penguins")
    # wandb.config["model_name"] = model.name
    wandb.config["model_config"] = model_config
    # wandb.config["model_info"] = model.info()
    wandb.config["preprocessors"] = [p.__class__.__name__ for p in PREPROCESSORS]

    challenge, _, _ = loader.load()
    models, evaluations = train(challenge)

    for i, metrics in enumerate(evaluations):
        wandb.log(metrics)
    wandb.log(pd.DataFrame(evaluations).add_prefix("mean_").mean().to_dict())

    # TODO: Ensemble the models

    # output = sorted(Path("AutogluonModels").glob("ag-*"), key=os.path.getmtime)[-1]
    # wandb.log({"raw_model_info": model.info()})
    # wandb.log_model(output, name="model")

    print("Done with training.")
