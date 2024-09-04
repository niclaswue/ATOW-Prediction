import pandas as pd
from utils.dataset import Dataset
from pathlib import Path

from preprocessing.base_preprocessor import BasePreprocessor

root_dir = Path(__file__).parent.parent.absolute()
base_dir = root_dir / "additional_data" / "airport_data"


class FuelPricePreprocessor(BasePreprocessor):

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding fuel price data...")
        fuel_df = pd.read_csv(
            base_dir / "fuel_prices_20_06_2022.csv", encoding="latin-1"
        )
        fuel_df = fuel_df[["Country", "Price Per Liter (USD)"]]

        un = pd.read_csv(base_dir / "UN_fuel_data.csv")
        un = un.rename(columns={"Country or Area": "Country"})
        un = un[un["Year"] == 2022]

        un = un.pivot_table(
            index="Country", columns=["Commodity - Transaction"], values=["Quantity"]
        )
        # TODO: Fix UN data => merging
        un.columns = ["_".join(col).strip() for col in un.columns.values]
        un = un.reset_index()

        fuel_df = pd.merge(fuel_df, un, on="Country", how="left")

        fuel_df = fuel_df.rename(columns={"Country": "name"})
        country_codes = pd.read_csv(base_dir / "country_codes.csv")
        country_codes = country_codes[["name", "alpha-2"]]
        fuel_df = pd.merge(fuel_df, country_codes, on="name").drop(columns="name")

        fuel_adep = fuel_df.rename(
            columns={
                "alpha-2": "country_code_adep",
                "Price Per Liter (USD)": "fuel_price_adep",
            }
        )
        dataset.df = pd.merge(dataset.df, fuel_adep, on="country_code_adep", how="left")

        fuel_ades = fuel_df.rename(
            columns={
                "alpha-2": "country_code_ades",
                "Price Per Liter (USD)": "fuel_price_ades",
            }
        )
        dataset.df = pd.merge(dataset.df, fuel_ades, on="country_code_ades", how="left")

        print("Done.")
        return dataset
