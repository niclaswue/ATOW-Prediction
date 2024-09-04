from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from models.base_model import BaseModel


class AutogluonModel(BaseModel):

    def __init__(
        self,
        time_limit=5 * 60,
        preset="high_quality",
        verbosity: int = 2,
        name: str = "autogluon",
    ):
        super().__init__(name)
        self.time_limit = time_limit
        self.presets = [preset]
        self.verbosity = verbosity

    def train(self, training_df: pd.DataFrame):
        train_data = TabularDataset(training_df)
        predictor = TabularPredictor(
            label="tow",
            verbosity=self.verbosity,
            # log_to_file=True,
            # log_file_path="/home/niclas/ATOW-Prediction/autogluon.log",
        ).fit(train_data, time_limit=self.time_limit, presets=self.presets)
        self.model = predictor

    def predict(self, input_df: pd.DataFrame):
        data = TabularDataset(input_df)
        y = self.model.predict(data)
        return y

    def info(self):
        return self.model.info()
