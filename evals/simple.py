import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


class SimpleEvals:

    @staticmethod
    def mae(gt: np.ndarray, pred: np.ndarray):
        return np.abs(gt - pred).mean()

    @staticmethod
    def mse(gt: np.ndarray, pred: np.ndarray):
        return (((gt - pred)) ** 2).mean()

    @staticmethod
    def r_squared(gt: np.ndarray, pred: np.ndarray):
        return r2_score(gt, pred)

    @staticmethod
    def mae_stddev(gt: np.ndarray, pred: np.ndarray):
        return np.abs(gt - pred).std()

    @staticmethod
    def max_abs_error(gt: np.ndarray, pred: np.ndarray):
        return np.abs(gt - pred).max()

    @staticmethod
    def relative_error(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6):
        return (gt - pred) / (gt + eps)

    @staticmethod
    def near(gt: np.ndarray, pred: np.ndarray, rel_tol: float = 0.05):
        # assign true to all values within 5% of the correct weight
        return SimpleEvals.relative_error(gt, pred) <= rel_tol

    def evaluate(self, ground_truth: pd.Series, predictions: pd.Series):
        gt = ground_truth.to_numpy()
        pred = predictions.to_numpy()

        return {
            "mae": SimpleEvals.mae(gt, pred),
            "mse": SimpleEvals.mse(gt, pred),
            "mae_stddev": SimpleEvals.mae_stddev(gt, pred),
            "max_abs_error": SimpleEvals.max_abs_error(gt, pred),
            "mean_relative_error": SimpleEvals.relative_error(gt, pred).mean(),
            "max_relative_error": SimpleEvals.relative_error(gt, pred).max(),
            "r_squared": SimpleEvals.r_squared(gt, pred),
            "percent_near": SimpleEvals.relative_error(gt, pred).mean(),
        }
