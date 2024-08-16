# This script can be used to execute the whole pipeline.
from pathlib import Path
from typing import List
import json

import pandas as pd

from utils.data_loader import DataLoader
from models.base_model import BaseModel
from models.average_model import AverageModel
from models.median_model import MedianModel
from evals.metrics import MetricEvals
from evals.compare_models import CompareModelsEval
from visualizations.compare_models import plot_metric_overview


EVALS = [MetricEvals()]
MODELS: List[BaseModel] = [AverageModel(), MedianModel()]


def main():
    loader = DataLoader(Path("data"), num_days=1, seed=1337)
    challenge, submission, trajectories = loader.load()
    train_df, val_df = challenge.split(train_percent=0.8)

    model_predictions = {}
    for model in MODELS:
        print(f"\n\nTraining model {model.name}")
        model.train(train_df)
        info = model.info()

        print("-" * 10)
        print(f"Model info: {json.dumps(info, indent=4)}")
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
