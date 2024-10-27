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

        # never seen in submission, only train data
        dataset.df = dataset.df[dataset.df["aircraft_type"] != "C56X"]
        dataset.df = dataset.df[dataset.df["aircraft_type"] != "A310"]

        # # drop columns not found to be important using feature importance analysis

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

        # 1. Drop cols with >90% missing
        missing_cols = dataset.df.columns[dataset.df.isnull().mean() > 0.9]
        dataset.df.drop(columns=missing_cols, inplace=True)

        # 2. Drop highly correlated features
        # corr_matrix = dataset.df.corr(numeric_only=True)
        # high_corr_features = []
        # for i in range(len(corr_matrix.columns)):
        #     for j in range(i):
        #         if abs(corr_matrix.iloc[i, j]) > 0.95:
        #             high_corr_features.append(corr_matrix.columns[i])
        #             break
        # dataset.df.drop(columns=high_corr_features, inplace=True)

        # 3. Drop low variance features
        # num_cols = dataset.df.select_dtypes(include=[np.number]).columns
        # low_var_cols = []
        # for col in num_cols:
        #     if dataset.df[col].std() < 0.001:  # Nearly constant
        #         low_var_cols.append(col)
        # dataset.df.drop(columns=low_var_cols, inplace=True)

        # 4. Basic missing value imputation
        # dataset.df = dataset.df.fillna(dataset.df.median())  # Numeric
        # dataset.df = dataset.df.fillna(dataset.df.mode().iloc[0])  # Categorical

        # # flight ids with more than 3% over MTOW are considered outliers
        # mtow_outliers = [
        #     249415582,
        #     250020264,
        #     250030428,
        #     250210254,
        #     250242828,
        #     251401350,
        #     252593101,
        #     253086784,
        #     253968219,
        #     254886542,
        #     255009398,
        #     256569887,
        #     257564559,
        #     256267626,
        #     257579742,
        #     249761274,
        #     257123965,
        #     257270731,
        #     254676967,
        #     254925394,
        #     255198787,
        #     254889763,
        #     249223866,
        #     249453575,
        #     249044014,
        #     256253559,
        #     256500632,
        #     257132947,
        #     257644354,
        #     249012236,
        #     249087461,
        #     249359448,
        #     249595670,
        #     250486323,
        #     250470528,
        #     250633561,
        #     250806675,
        #     249900202,
        #     250803601,
        #     250833861,
        #     251215615,
        #     251773655,
        #     251788228,
        #     257606134,
        #     257861616,
        #     249746404,
        #     249810526,
        #     249988920,
        #     256647504,
        #     256756597,
        #     256796886,
        #     257125186,
        #     257251064,
        #     257336505,
        #     257380942,
        #     257494990,
        #     257516707,
        #     248793831,
        #     255578568,
        #     249734617,
        #     249748175,
        #     249900837,
        #     250079825,
        #     253972052,
        #     248981315,
        #     249649454,
        #     256800040,
        #     254286130,
        #     255473710,
        #     256328774,
        #     249685841,
        #     252571156,
        #     252993039,
        #     256374063,
        #     256616550,
        #     256897998,
        #     249730980,
        #     249211733,
        #     249231270,
        #     249614081,
        #     249623674,
        #     249691008,
        #     250539386,
        #     255036300,
        #     255159827,
        #     255171147,
        #     255309972,
        #     255445381,
        #     256355387,
        #     257068568,
        #     257443028,
        #     258018268,
        #     258072494,
        #     256562782,
        #     257294883,
        #     249467026,
        #     256433193,
        #     257067655,
        #     258069856,
        #     250119981,
        # ]
        # dataset.df = dataset.df[~dataset.df["flight_id"].isin(mtow_outliers)]

        return dataset
