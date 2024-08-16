import pandas as pd
from models.base_model import BaseModel


class MedianModel(BaseModel):
    def train(self, training_df: pd.DataFrame):
        self.median = training_df.tow.median()

    def predict(self, input_df: pd.DataFrame):
        return pd.Series([self.median] * len(input_df))

    def info(self):
        return {"median_value": self.median}
