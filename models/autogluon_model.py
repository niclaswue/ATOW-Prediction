from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from models.base_model import BaseModel


class AutogluonModel(BaseModel):

    def __init__(self, name: str = "autogluon"):
        super().__init__(name)

    def train(self, training_df: pd.DataFrame):
        train_data = TabularDataset(training_df)
        predictor = TabularPredictor(label="tow").fit(train_data)
        self.model = predictor

    def predict(self, input_df: pd.DataFrame):
        data = TabularDataset(input_df)
        y = self.model.predict(data)
        return y

    def info(self):
        return {}
