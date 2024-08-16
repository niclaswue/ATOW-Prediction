import pandas as pd
from models.base_model import BaseModel


class AverageModel(BaseModel):
    def train(self, training_df: pd.DataFrame):
        self.average = training_df.tow.mean()

    def predict(self, input_df: pd.DataFrame):
        return pd.Series([self.average] * len(input_df))

    def info(self):
        return {"mean_value": self.average}
