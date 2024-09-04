import pandas as pd
from utils.dataset import Dataset
from tqdm import tqdm
from functools import cache
from pathlib import Path

from preprocessing.base_preprocessor import BasePreprocessor

base_dir = Path("additional_data/airport_data/")


class PaxFlowPreprocessor(BasePreprocessor):

    @cache
    def load_statistics(self):
        print("Loading statistics..")
        # https://ec.europa.eu/eurostat/cache/metadata/en/avia_pa_esms.htm
        sdf = pd.read_csv(base_dir / "estat_avia_tf_apal_en.csv")
        sdf.drop(columns=["OBS_FLAG", "DATAFLOW", "LAST UPDATE", "freq"], inplace=True)

        sdf["TIME_PERIOD"] = sdf["TIME_PERIOD"].astype(str)
        sdf["rep_airp"] = sdf["rep_airp"].astype(str)

        # to make it faster, if another year add condition or uncomment
        sdf = sdf[sdf["TIME_PERIOD"].str.startswith("2022-", na=False)]
        return sdf

    def _get_statistic_data(self, year, month, airport):
        sdf = self.load_statistics()
        mask = (sdf["rep_airp"] == airport) & (
            sdf["TIME_PERIOD"] == f"{year}-{month:0>2}"
        )
        stats = sdf[mask]
        if len(stats) == 0:
            # print(f"NO data for AP: {airport} at {year}-{month}")
            return stats

        df_pivot = stats.pivot_table(
            index=None,
            columns=["unit", "tra_meas", "airline"],
            values="OBS_VALUE",
        )

        df_pivot.columns = [
            "{}_{}_{}".format(col[0], col[1], col[2]) for col in df_pivot.columns
        ]

        df_pivot.reset_index(drop=True, inplace=True)
        return df_pivot

    def process(self, dataset: Dataset) -> Dataset:
        sdf = self.load_statistics()
        # Extract year and month once
        dataset.df["year"] = pd.to_datetime(dataset.df["date"]).dt.year
        dataset.df["month"] = pd.to_datetime(dataset.df["date"]).dt.month

        # Create full airport identifiers
        dataset.df["full_adep"] = (
            dataset.df["country_code_adep"] + "_" + dataset.df["adep"]
        )
        dataset.df["full_ades"] = (
            dataset.df["country_code_ades"] + "_" + dataset.df["ades"]
        )

        # Filter relevant airports
        relevant_airports = sdf.rep_airp.unique()
        filtered_df = dataset.df[
            dataset.df["full_adep"].isin(relevant_airports)
            | dataset.df["full_ades"].isin(relevant_airports)
        ]

        # Create a DataFrame for all combinations of airport, year, and month
        unique_airports = pd.concat(
            [filtered_df["full_adep"], filtered_df["full_ades"]]
        ).unique()
        unique_years = filtered_df["year"].unique()
        unique_months = filtered_df["month"].unique()

        # Create all combinations
        combinations = pd.MultiIndex.from_product(
            [unique_airports, unique_years, unique_months],
            names=["airport", "year", "month"],
        ).to_frame(index=False)

        # Fetch statistics data for each combination and aggregate
        stats_dfs = []
        total = len(combinations)
        for ap, y, m in tqdm(
            combinations.itertuples(index=False, name=None), total=total
        ):
            stats = self._get_statistic_data(y, m, ap)
            if len(stats) == 0:
                continue
            stats = stats.add_prefix(f"stats_")
            stats["full_airport"] = ap
            stats["year"] = y
            stats["month"] = m
            stats_dfs.append(stats)

        if stats_dfs:
            stats_df = pd.concat(stats_dfs, ignore_index=True)

            # Merge statistics back into the original dataframe
            for mode in ["adep", "ades"]:
                dataset.df = dataset.df.merge(
                    stats_df,
                    how="left",
                    left_on=[f"full_{mode}", "year", "month"],
                    right_on=["full_airport", "year", "month"],
                )
                dataset.df.drop(columns=["full_airport"], inplace=True)

        return dataset
