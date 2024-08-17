import pandas as pd
from typing import List
from models.base_model import BaseModel
import numpy as np


class EnsembleModel(BaseModel):
    def __init__(self, model_list: List[BaseModel], name: str = None):
        name = name or f"Ens({[m.name for m in model_list]})"
        super().__init__(name)
        self.models = model_list

    def train(self, training_df: pd.DataFrame):
        for model in self.models:
            model.train(training_df)

    def predict(self, input_df: pd.DataFrame):
        results = []
        for model in self.models:
            results.append(model.predict(input_df).to_numpy())
        results = np.stack(results).mean(axis=0)
        return pd.Series(results)

    def info(self):
        return {m.name: m.info() for m in self.models}
