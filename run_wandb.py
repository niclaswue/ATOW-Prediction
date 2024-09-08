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

from models.autogluon_model import AutogluonModel
from evals.metrics import MetricEvals
import argparse

try:
    import wandb

    WANDB = True
except ImportError:
    WANDB = False

pd.set_option("future.no_silent_downcasting", True)
warnings.filterwarnings(action="ignore", message="Mean of empty slice")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--quality", type=str, help="Quality Preset", default="high_quality"
)
parser.add_argument("--time", type=int, help="Time Limit (s)", default=300)
args = parser.parse_args()

PREPROCESSORS: List[BasePreprocessor] = [
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
model = AutogluonModel(**model_config)
loader = DataLoader(Path("data"), num_days=0)


def train():
    challenge, submission, final_submission = loader.load()
    for preprocessor in PREPROCESSORS:
        challenge = preprocessor.apply(challenge)

    train_df, val_df = challenge.split(train_percent=0.8, seed=0)
    print(f"\n\nTraining model {model.name}")
    model.train(train_df)
    predictions = model.predict(val_df)
    evaluator.log_evaluation(val_df.tow, predictions)

    # Only works for autogluon at the moment
    model.log_feature_importance(train_df)

    return model


def logging_wandb_init():
    wandb.init(project="flying_penguins")
    wandb.config["model_name"] = model.name
    wandb.config["model_config"] = model_config
    wandb.config["model_info"] = model.info()
    wandb.config["preprocessors"] = [p.__class__.__name__ for p in PREPROCESSORS]


def logging_wandb_post_training(model):
    output = sorted(Path("AutogluonModels").glob("ag-*"), key=os.path.getmtime)[-1]
    wandb.log({"raw_model_info": model.info()})
    wandb.log_model(output)


if __name__ == "__main__":
    if WANDB:
        logging_wandb_init()

    model = train()

    if WANDB:
        logging_wandb_post_training(model)

    print("Done with training.")