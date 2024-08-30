# This script can be used to execute the whole pipeline.
from pathlib import Path
from typing import List
from pprint import pprint

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost

from utils.data_loader import DataLoader
from preprocessing.aircraft_performance import add_aircraft_performance_data
from preprocessing.weather import add_weather_data
from preprocessing.runway import add_runway_data
from preprocessing.statistics import add_statistics_data
from models.base_model import BaseModel
from models.average_model import AverageModel
from models.median_model import MedianModel
from models.ensemble import EnsembleModel
from models.scikit_learn_model import ScikitLearnModel
from evals.metrics import MetricEvals
from evals.compare_models import CompareModelsEval
from visualizations.compare_models import plot_metric_overview


rf_model = ScikitLearnModel(RandomForestRegressor, {"n_estimators": 1, "verbose": 5})
xgb = ScikitLearnModel(xgboost.XGBRegressor)
# knn = ScikitLearnModel(KNeighborsRegressor, {"n_neighbors": 10, "weights": "distance"})
ensemble = EnsembleModel([xgb, rf_model])

EVALS = [MetricEvals()]
MODELS: List[BaseModel] = [xgb]


def main():
    loader = DataLoader(Path("data"), num_days=1, seed=1337)
    challenge, submission, final_submission, trajectories = loader.load()

    # challenge.df = pd.read_parquet("preprocessed_latest.parquet")

    challenge = add_statistics_data(challenge)
    challenge = add_weather_data(challenge)
    challenge = add_aircraft_performance_data(challenge)
    challenge = add_runway_data(challenge)

    challenge.df.to_parquet("preprocessed_latest.parquet")

    datetime_cols = ["date", "actual_offblock_time", "arrival_time", "valid"]
    for c in datetime_cols:
        if c not in challenge.df.columns:
            continue
        challenge.df[c] = pd.to_datetime(challenge.df[c]).astype(int)

    train_df, val_df = challenge.split(train_percent=0.8)

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
