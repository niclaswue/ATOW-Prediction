import openap.prop
import openap
from tqdm import tqdm
from functools import cache
from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset
import json
from pathlib import Path
import re
import pandas as pd
import yaml


class OpenAPAircraftPerformancePreprocessor(BasePreprocessor):
    def __init__(self, no_cache=False) -> None:
        super().__init__(no_cache)
        self.base_path = (
            Path(__file__).parent.parent / "additional_data" / "aircraft_data"
        )

    @cache
    def props_for_aircraft(self, aircraft_type: str) -> dict:
        try:
            props = openap.prop.aircraft(aircraft_type)
        except ValueError as e:
            manual_info = self.base_path / f"{aircraft_type.lower()}.yaml"
            if manual_info.exists():
                with open(manual_info, "r") as f:
                    props = yaml.safe_load(f)
            elif aircraft_type == "C56X":
                return {}  # not important
            else:
                raise ValueError(f"Unknown type {aircraft_type}")

        try:
            engine = props.get("engine", {}).get("default")
            engine_props = openap.prop.engine(engine)
        except ValueError as e:
            manual_info = self.base_path / f"{engine}.yaml"
            if manual_info.exists():
                with open(manual_info, "r") as f:
                    engine_props = yaml.safe_load(f)
            else:
                raise e

        props = pd.json_normalize(props)
        engine_props = pd.json_normalize(engine_props)
        all_data = pd.concat((props, engine_props), axis=1)
        options_cols = [c for c in all_data.columns if c.startswith("engine.options")]
        # duplicated limits
        limits_cols = [c for c in all_data.columns if c.startswith("limits")]
        all_data.drop(columns=options_cols, inplace=True)
        all_data.drop(columns=limits_cols, inplace=True)
        all_data.drop(columns=["uid"], inplace=True)
        return all_data.iloc[0].to_dict()

    def process(self, dataset: Dataset) -> Dataset:
        # tqdm.pandas()

        # combined column
        dataset.df["airline_aircraft"] = (
            dataset.df["airline"] + "_" + dataset.df["aircraft_type"]
        )

        cols = self.props_for_aircraft(dataset.df["aircraft_type"].iloc[0]).keys()
        for col in cols:
            dataset.df[f"openap_{col}"] = dataset.df["aircraft_type"].progress_apply(
                lambda x: self.props_for_aircraft(x).get(col)
            )
        return dataset


# TODO: Add icao engine data
