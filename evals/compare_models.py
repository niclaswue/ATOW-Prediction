import pandas as pd
import numpy as np
from typing import Dict


class CompareModelsEval:

    @staticmethod
    def win_rate(gt, preds, model_names: list):
        closest_idx = np.abs((preds - gt)).argmin(axis=0)
        _, vals = np.unique(closest_idx, return_counts=True)
        win_rate = vals / len(gt)
        return {name: {"win_rate": wr} for name, wr in zip(model_names, win_rate)}

    def evaluate(self, ground_truth: pd.Series, predictions: Dict[str, pd.Series]):
        gt = ground_truth.to_numpy()
        preds = [p.to_numpy() for p in predictions.values()]
        preds = np.stack(preds)
        win_rate = CompareModelsEval.win_rate(gt, preds, predictions.keys())
        return win_rate
