import os
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from models.base_model import BaseModel
from pprint import pprint
import wandb


class AutogluonModel(BaseModel):
    def __init__(
        self,
        time_limit=5 * 60,
        preset="high_quality",
        verbosity: int = 2,
        name: str = "autogluon",
        wandb: bool = True,
    ):
        super().__init__(name)
        self.time_limit = time_limit
        self.presets = [preset]
        self.verbosity = verbosity
        self.wandb = wandb

    def train(self, training_df: pd.DataFrame):
        num_cpus = min(os.cpu_count(), 64)

        training_df.drop(columns=["flight_id"], inplace=True)
        train_data = TabularDataset(training_df)
        predictor = TabularPredictor(
            label="tow",
            verbosity=self.verbosity,
            log_to_file=True,
        ).fit(
            train_data,
            time_limit=self.time_limit,
            presets=self.presets,
            num_cpus=num_cpus,
        )
        self.model = predictor

    def log_feature_importance(self, train_data: pd.DataFrame):
        importance = self.model.feature_importance(
            train_data, time_limit=self.time_limit * 0.1
        )
        if self.wandb:
            importance = importance.reset_index()
            importance_table = wandb.Table(dataframe=importance)
            importance_artifact = wandb.Artifact("feature_importance", type="dataset")
            importance_artifact.add(importance_table, "importance_table")
            wandb.run.log({"feature_importance": importance_table})
            wandb.run.log_artifact(importance_artifact)
        pprint(importance.to_dict())

    def predict(self, input_df: pd.DataFrame):
        flight_ids = input_df["flight_id"]
        input_df.drop(columns=["flight_id"], inplace=True)
        data = TabularDataset(input_df)
        y = self.model.predict(data)

        if self.wandb:
            input_df["prediction"] = y
            if "tow" not in input_df.columns:
                input_df["tow"] = 0
            pred = input_df[["prediction", "tow"]]
            pred["flight_id"] = flight_ids
            pred["error"] = input_df["prediction"] - input_df["tow"]

            pred_table = wandb.Table(dataframe=pred)
            prediction_artifact = wandb.Artifact("predictions", type="dataset")
            prediction_artifact.add(pred_table, "pred_table")

            wandb.run.log({"predictions": pred_table})
            wandb.run.log_artifact(prediction_artifact)
        return y

    def info(self):
        return {
            "time_limit": self.time_limit,
            "presets": self.presets,
        }
