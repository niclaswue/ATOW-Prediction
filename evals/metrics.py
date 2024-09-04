import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


class MetricEvals:

    @staticmethod
    def mae(gt: np.ndarray, pred: np.ndarray):
        return np.abs(gt - pred).mean()

    @staticmethod
    def mse(gt: np.ndarray, pred: np.ndarray):
        return (((gt - pred)) ** 2).mean()

    @staticmethod
    def rmse(gt: np.ndarray, pred: np.ndarray):
        return np.sqrt((((gt - pred)) ** 2).mean())

    @staticmethod
    def mape(gt: np.ndarray, pred: np.ndarray):
        return np.mean(np.abs((gt - pred) / gt)) * 100

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
        return np.abs((gt - pred)) / (gt + eps)

    @staticmethod
    def near(gt: np.ndarray, pred: np.ndarray, rel_tol: float = 0.10):
        # assign true to all values within 10% of the correct weight
        return MetricEvals.relative_error(gt, pred) <= rel_tol

    def evaluate(self, ground_truth: pd.Series, predictions: pd.Series):
        gt = ground_truth.to_numpy()
        pred = predictions.to_numpy()

        return {
            "rmse (↓)": MetricEvals.rmse(gt, pred),
            "MAPE (↓)": MetricEvals.mape(gt, pred),
            "mae (↓)": MetricEvals.mae(gt, pred),
            "mse (↓)": MetricEvals.mse(gt, pred),
            "mae_stddev (↓)": MetricEvals.mae_stddev(gt, pred),
            "max_abs_error (↓)": MetricEvals.max_abs_error(gt, pred),
            "mean_relative_error (↓)": MetricEvals.relative_error(gt, pred).mean(),
            "max_relative_error (↓)": MetricEvals.relative_error(gt, pred).max(),
            "r_squared (↑)": MetricEvals.r_squared(gt, pred),
            "percent_near (↑)": 100 * MetricEvals.relative_error(gt, pred).mean(),
        }

    def print_evaluation(self, ground_truth: pd.Series, predictions: pd.Series):
        evaluation = self.print_evaluation(ground_truth, predictions)
        for k, v in evaluation.items():
            print(f"{k:<20}: {v:.3f}")
