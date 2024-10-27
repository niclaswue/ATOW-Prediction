from preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
import numpy as np
from utils.dataset import Dataset


class WeatherSafetyFeatures(BasePreprocessor):
    def __init__(self, no_cache=False) -> None:
        super().__init__(no_cache)
        # Constants for weather safety thresholds
        self.VISIBILITY_MIN = 5000  # meters (about 3 statute miles)
        self.WIND_GUST_THRESHOLD = 20  # knots
        self.CROSSWIND_MAX = 25  # knots
        self.CEILING_MIN = 1500  # feet - typical approach minimum
        self.ICE_THRESHOLD = 0.1  # any ice accretion above this is significant

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding weather safety features...")

        df = dataset.df

        # Visibility conditions
        self._add_visibility_features(df)

        # Wind conditions
        self._add_wind_features(df)

        # Ceiling and approach conditions
        self._add_ceiling_features(df)

        # Temperature and icing conditions
        self._add_temperature_features(df)

        # Combined risk factors
        self._add_combined_weather_risks(df)

        print("Done.")
        return dataset

    def _add_visibility_features(self, df):
        """Features related to visibility conditions"""
        # Convert visibility to meters if needed (assuming vsby is in statute miles)
        df["visibility_meters"] = df["ades_vsby"] * 1609.34

        # Low visibility conditions
        df["is_low_visibility"] = df["visibility_meters"] < self.VISIBILITY_MIN

        # Rate visibility conditions
        df["visibility_severity"] = pd.cut(
            df["visibility_meters"],
            bins=[0, 800, 1600, 3200, 5000, float("inf")],
            labels=[4, 3, 2, 1, 0],  # Higher number = more severe
        ).fillna(0)

    def _add_wind_features(self, df):
        """Features related to wind conditions"""
        # Basic wind features
        df["has_significant_gusts"] = df["ades_gust"] > self.WIND_GUST_THRESHOLD

        # Analyze peak winds
        df["has_strong_peak_winds"] = (
            df["ades_peak_wind_gust"] > self.WIND_GUST_THRESHOLD
        )

        # Crosswind component calculation
        def calculate_crosswind(row):
            if pd.isna(row["ades_sknt"]) or pd.isna(row["ades_drct"]):
                return 0
            # Assuming runway heading is available, otherwise use approximation
            runway_heading = 360  # This should be replaced with actual runway heading
            wind_angle = abs(runway_heading - row["ades_drct"])
            return abs(row["ades_sknt"] * np.sin(np.radians(wind_angle)))

        df["crosswind_component"] = df.apply(calculate_crosswind, axis=1)
        df["has_critical_crosswind"] = df["crosswind_component"] > self.CROSSWIND_MAX

        # Wind variability
        df["wind_variability"] = np.where(
            df["ades_gust"].notna(), df["ades_gust"] - df["ades_sknt"], 0
        )

        # Overall wind severity
        df["wind_severity"] = (
            (df["has_significant_gusts"].astype(int) * 2)
            + (df["has_strong_peak_winds"].astype(int) * 2)
            + (df["has_critical_crosswind"].astype(int) * 3)
        )

    def _add_ceiling_features(self, df):
        """Features related to ceiling and approach conditions"""
        # Get lowest ceiling from available layers
        ceiling_columns = ["ades_skyl1", "ades_skyl2", "ades_skyl3", "ades_skyl4"]
        df["lowest_ceiling"] = df[ceiling_columns].replace(0, np.nan).min(axis=1)

        # Ceiling conditions
        df["is_low_ceiling"] = df["lowest_ceiling"] < self.CEILING_MIN

        # Approach category based on ceiling and visibility
        df["approach_severity"] = np.where(
            (df["lowest_ceiling"] < 500) | (df["visibility_meters"] < 1600),
            3,  # CAT III conditions
            np.where(
                (df["lowest_ceiling"] < 800) | (df["visibility_meters"] < 3200),
                2,  # CAT II conditions
                np.where(
                    (df["lowest_ceiling"] < self.CEILING_MIN)
                    | (df["visibility_meters"] < self.VISIBILITY_MIN),
                    1,  # CAT I conditions
                    0,  # Visual conditions
                ),
            ),
        )

    def _add_temperature_features(self, df):
        """Features related to temperature and icing conditions"""
        # Temperature spread (dewpoint depression)
        df["temp_dewpoint_spread"] = df["ades_tmpf"] - df["ades_dwpf"]

        # Icing conditions
        df["has_reported_icing"] = (
            (df["ades_ice_accretion_1hr"] > self.ICE_THRESHOLD)
            | (df["ades_ice_accretion_3hr"] > self.ICE_THRESHOLD)
            | (df["ades_ice_accretion_6hr"] > self.ICE_THRESHOLD)
        )

        # Potential icing conditions (temperature range where icing is possible)
        df["icing_conditions_likely"] = (
            (df["ades_tmpf"].between(-10, 2))  # Temperature range for icing
            & (df["temp_dewpoint_spread"] < 3)  # High humidity
        )

        # Snow conditions
        df["has_snow"] = df["ades_snowdepth"] > 0

    def _add_combined_weather_risks(self, df):
        """Combined weather risk assessment"""
        # Individual risk factors
        risk_factors = [
            ("visibility_risk", df["visibility_severity"] > 0),
            ("wind_risk", df["wind_severity"] > 2),
            ("ceiling_risk", df["is_low_ceiling"]),
            ("icing_risk", df["has_reported_icing"] | df["icing_conditions_likely"]),
            ("snow_risk", df["has_snow"]),
        ]

        # Count total risk factors
        df["total_weather_risks"] = sum(risk[1].astype(int) for risk in risk_factors)

        # Overall weather severity classification
        df["weather_severity"] = pd.cut(
            df["total_weather_risks"],
            bins=[-np.inf, 0, 1, 2, np.inf],
            labels=["normal", "caution", "severe", "critical"],
        )

        # Fuel planning impact
        df["extra_fuel_recommended"] = df["total_weather_risks"] >= 2

        # Specific fuel factor based on conditions
        df["weather_fuel_factor"] = (
            1.0  # Base factor
            + (
                pd.to_numeric(df["visibility_severity"]) * 0.02
            )  # Each visibility severity level adds 2%
            + (df["wind_severity"] * 0.015)  # Each wind severity level adds 1.5%
            + (df["approach_severity"] * 0.03)  # Each approach severity level adds 3%
            + (df["has_reported_icing"].astype(int) * 0.05)  # Icing adds 5%
            + (df["has_snow"].astype(int) * 0.03)  # Snow adds 3%
        )

        # Diversion risk assessment
        df["diversion_risk"] = (
            (df["visibility_severity"] >= 3)
            | (df["wind_severity"] >= 4)
            | (df["approach_severity"] >= 2)
            | (df["has_reported_icing"])
            | (df["has_snow"] & (df["ades_snowdepth"] > 2))
        )
