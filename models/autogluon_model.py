from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from models.base_model import BaseModel


class AutogluonModel(BaseModel):

    def __init__(self, time_limit=5 * 60, name: str = "autogluon"):
        super().__init__(name)
        self.time_limit = time_limit

    def train(self, training_df: pd.DataFrame):
        train_data = TabularDataset(training_df)
        predictor = TabularPredictor(
            label="tow",
            verbosity=0,
            log_to_file=True,
            log_file_path="/home/niclas/ATOW-Prediction/autogluon.log",
        ).fit(train_data, time_limit=self.time_limit, presets=["high_quality"])
        self.model = predictor

    def predict(self, input_df: pd.DataFrame):
        data = TabularDataset(input_df)
        y = self.model.predict(data)
        return y

    def info(self):
        return {}
