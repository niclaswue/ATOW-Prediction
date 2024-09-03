# This script can be used to execute the whole pipeline.
from pathlib import Path
from typing import List, Callable
from pprint import pprint
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost
import lightgbm as lgb

from utils.data_loader import DataLoader

from preprocessing.clean_dataset import clean_dataset
from preprocessing.aircraft_performance import add_aircraft_performance_data
from preprocessing.weather import add_weather_data
from preprocessing.runway import add_runway_data
from preprocessing.airport_information import (
    add_airport_pax_flow,
    add_fuel_price_data,
)
from preprocessing.derived_features import add_derived_features
from preprocessing.trajectory_features import add_trajectory_features

# from preprocessing.augment_features import augment_features
from models.base_model import BaseModel
from models.ensemble import EnsembleModel
from models.autogluon_model import AutogluonModel
from models.scikit_learn_model import ScikitLearnModel
from evals.metrics import MetricEvals
from evals.compare_models import CompareModelsEval
from visualizations.compare_models import plot_metric_overview

warnings.filterwarnings(action="ignore", message="Mean of empty slice")

# linear regression ensembling catboost, lightgbm, autogluon, xgboost with openFE and feature importance selection
rf_model = ScikitLearnModel(RandomForestRegressor, {"n_estimators": 10, "verbose": 5})
xgb = ScikitLearnModel(xgboost.XGBRegressor)
ag = AutogluonModel(time_limit=5 * 60)

ensemble = EnsembleModel([xgb, rf_model])

EVALS = [MetricEvals()]
MODELS: List[BaseModel] = [ag]
FEATURES: List[Callable] = [
    add_fuel_price_data,
    add_airport_pax_flow,
    add_derived_features,
    add_aircraft_performance_data,
    add_runway_data,
    add_weather_data,
    clean_dataset,
]
# add_trajectory_features
#


def main():
    loader = DataLoader(Path("data"), num_days=1, seed=1337)
    challenge, submission, final_submission, trajectories = loader.load()

    for preprocessing_func in FEATURES:
        challenge = preprocessing_func(challenge)

    train_df, val_df = challenge.split(train_percent=0.8)
    # train_df, val_df = augment_features(train_df, val_df, y="tow")

    model_predictions = {}
    for model in MODELS:
        print(f"\n\nTraining model {model.name}")
        model.train(train_df)
        info = model.info()

        print("-" * 10)
        print(f"Model info: {pprint(info)}")
        print("-" * 10)
        predictions = model.predict(val_df)
        model_predictions[model.name] = predictions

    # evaluate the predictions per model
    model_evals = {}
    for model_name, predictions in model_predictions.items():
        for evaluator in EVALS:
            evaluation = evaluator.evaluate(val_df.tow, predictions)
            model_evals[model_name] = evaluation
            for k, v in evaluation.items():
                print(f"{k:<20}: {v:.3f}")
            print("-" * 10)

    # compare the models
    cme = CompareModelsEval()
    compared = cme.evaluate(val_df.tow, model_predictions)

    model_evals = pd.DataFrame(model_evals)
    compared = pd.DataFrame(compared)
    model_evals = pd.concat((model_evals, compared))
    plot_metric_overview(model_evals)


if __name__ == "__main__":
    main()
