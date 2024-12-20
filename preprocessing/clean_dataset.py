from utils.dataset import Dataset
import pandas as pd
import numpy as np
from preprocessing.base_preprocessor import BasePreprocessor


class CleanDatasetPreprocessor(BasePreprocessor):
    def process(self, dataset: Dataset) -> Dataset:
        # TODO: Remove duplicates in data etc.
        datetime_cols = ["date", "actual_offblock_time", "arrival_time", "valid"]
        for c in datetime_cols:
            if c not in dataset.df.columns:
                continue
            dataset.df[c] = pd.to_datetime(dataset.df[c])

        # very few samples for these aircraft types, we remove them
        dataset.df = dataset.df[dataset.df["aircraft_type"] != "C56X"]
        dataset.df = dataset.df[dataset.df["aircraft_type"] != "A310"]

        # drop columns found to be least important using feature importance analysis
        # this leads to a speed up during training
        drop_cols = [
            "Cost Index",
            "ades_mslp",
            "stats_PAS_PAS_TRF_UNK_x",
            "Quantity_Kerosene-type Jet Fuel - Consumption by manufacturing, construction and non-fuel mining industry_x",
            "stats_PAS_PAS_TRF_UNK_y",
            "ades_p01i",
            "stats_MOVE_CACM_LIC_NEU_y",
            "ades_skyl4",
            "stats_T_FRM_LD_NLD_UNK_y",
            "stats_PAS_PAS_TRS_UNK_x",
            "stats_PAS_PAS_TRF_LIC_EU_y",
            "stats_MOVE_CACM_UNK_y",
            "stats_PAS_PAS_TRS_UNK_y",
            "stats_PAS_PAS_CRD_LIC_EU_y",
            "stats_PAS_PAS_TRF_LIC_NEU_y",
            "stats_PAS_PAS_CRD_LIC_EU_x",
            "ades_continent",
            "stats_MOVE_ACM_LIC_NEU_y",
            "ades_skyc4",
            "stats_T_FRM_LD_NLD_LIC_EU_y",
            "stats_T_FRM_LD_NLD_UNK_x",
            "stats_MOVE_CACM_UNK_x",
            "Quantity_Kerosene-type Jet Fuel - Consumption by other_y",
            "stats_T_FRM_LD_NLD_LIC_NEU_y",
            "stats_T_FRM_LD_NLD_LIC_EU_x",
            "stats_MOVE_ACM_UNK_y",
            "stats_MOVE_CACM_LIC_NEU_x",
            "stats_MOVE_ACM_LIC_EU_x",
            "stats_PAS_PAS_TRF_LIC_NEU_x",
            "ades_skyc3",
            "stats_MOVE_ACM_UNK_x",
            "stats_MOVE_CACM_LIC_EU_y",
            "stats_MOVE_ACM_LIC_NEU_x",
            "stats_PAS_PAS_CRD_UNK_y",
            "Quantity_Kerosene-type Jet Fuel - Final consumption_y",
            "stats_PAS_PAS_CRD_LIC_NEU_y",
            "Quantity_Kerosene-type Jet Fuel - Consumption by other_x",
            "stats_PAS_PAS_TRF_LIC_EU_x",
            "stats_PAS_PAS_CRD_UNK_x",
            "stats_PAS_PAS_CRD_LIC_NEU_x",
            "Quantity_Kerosene-type Jet Fuel - Exports_y",
            "Quantity_Kerosene-type Jet Fuel - Consumption by domestic aviation_y",
            "stats_PAS_PAS_TRS_LIC_NEU_y",
            "stats_PAS_PAS_TRS_LIC_NEU_x",
            "stats_PAS_PAS_TRS_LIC_EU_y",
            "stats_T_FRM_LD_NLD_LIC_NEU_x",
            "stats_MOVE_ACM_LIC_EU_y",
            "stats_MOVE_CACM_LIC_EU_x",
            "stats_PAS_PAS_TRS_LIC_EU_x",
        ]

        dataset.df.drop(columns=drop_cols, inplace=True)

        # Drop cols with >90% missing
        missing_cols = dataset.df.columns[dataset.df.isnull().mean() > 0.9]
        dataset.df.drop(columns=missing_cols, inplace=True)

        # Separate numeric and categorical columns
        numeric_cols = dataset.df.select_dtypes(
            include=["int64", "float64", "float32"]
        ).columns

        # 3. Handle infinite values in numeric columns
        print("removing NaNs")
        for col in numeric_cols:
            # Replace inf with nan
            dataset.df[col] = dataset.df[col].replace([np.inf, -np.inf], np.nan)
        return dataset
