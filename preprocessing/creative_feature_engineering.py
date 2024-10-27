from preprocessing.base_preprocessor import BasePreprocessor
import pandas as pd
import numpy as np
from utils.dataset import Dataset


class CreativeWeightPreprocessor(BasePreprocessor):
    def __init__(self, no_cache=False) -> None:
        super().__init__(no_cache)
        self.business_hours = set(range(9, 18))
        self.meal_times = {
            "breakfast": set(range(6, 9)),
            "lunch": set(range(11, 14)),
            "dinner": set(range(18, 21)),
        }

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding creative experimental features...")

        df = dataset.df

        self._add_business_travel_features(df)
        self._add_physics_inspired_features(df)
        self._add_psychological_pricing_features(df)
        self._add_competition_features(df)
        self._add_environmental_impact_features(df)
        self._add_operational_complexity_features(df)
        self._add_passenger_behavior_features(df)

        print("Done.")
        return dataset

    def _add_business_travel_features(self, df):
        """Features based on business travel patterns"""
        # Time-based business travel indicators
        df["departure_hour"] = pd.to_datetime(df["actual_offblock_time"]).dt.hour
        df["is_business_hour"] = df["departure_hour"].isin(self.business_hours)

        # Business route patterns
        df["is_business_route"] = (
            (df["Seats Business_Class"] > 0)
            & (
                df["flown_distance"].between(200, 2000)
            )  # Typical business route distances
            & (df["is_business_hour"])
        )

        # Financial district routes
        major_financial_cities = ["London", "Frankfurt", "Paris", "Amsterdam", "Zurich"]
        df["is_financial_route"] = df["adep_city"].isin(major_financial_cities) | df[
            "ades_city"
        ].isin(major_financial_cities)

        # Weekend effect
        df["is_monday_morning"] = (df["day_of_week"] == 0) & (df["departure_hour"] < 10)
        df["is_friday_evening"] = (df["day_of_week"] == 4) & (df["departure_hour"] > 16)

    def _add_physics_inspired_features(self, df):
        """Features inspired by physics and aerodynamics"""
        # Bernoulli-inspired features
        if all(col in df.columns for col in ["openap_wing.area", "mean_cruise_speed"]):
            df["bernoulli_coefficient"] = (df["mean_cruise_speed"] ** 2) * df[
                "openap_wing.area"
            ]

        # Energy state approximation
        df["potential_energy_factor"] = df["cruise_altitude"] * df["MTOW"]
        if "mean_cruise_speed" in df.columns:
            df["kinetic_energy_factor"] = (
                0.5 * df["MTOW"] * (df["mean_cruise_speed"] ** 2)
            )

        # Reynolds number approximation (simplified)
        df["reynolds_factor"] = df["openap_fuselage.length"] * df["mean_cruise_speed"]

        # Thrust-to-drag estimation
        if "openap_max_thrust" in df.columns:
            df["thrust_drag_ratio"] = (
                df["openap_max_thrust"]
                * df["openap_engine.number"]
                / (df["openap_drag.cd0"] * df["openap_wing.area"])
            )

    def _add_psychological_pricing_features(self, df):
        """Features based on airline pricing psychology"""
        # Meal service likelihood
        df["meal_service_expected"] = df["departure_hour"].apply(
            lambda x: any(x in meal_time for meal_time in self.meal_times.values())
        )

        # Premium time slots
        df["is_premium_hour"] = (
            df["departure_hour"].between(7, 9)
        ) | (  # Morning business
            df["departure_hour"].between(16, 19)
        )  # Evening business

        # Holiday period detection
        df["is_holiday_month"] = df["month"].isin([7, 8, 12])  # Summer and Christmas

        # Weekend getaway pattern
        df["is_weekend_getaway"] = (
            (df["day_of_week"] == 4) & (df["departure_hour"] > 15)
        ) | (  # Friday evening
            (df["day_of_week"] == 6) & (df["departure_hour"] < 12)
        )  # Sunday morning

    def _add_competition_features(self, df):
        """Features based on competition and market dynamics"""
        # Route popularity (based on stats if available)
        if "stats_PAS_PAS_CRD_TOTAL_y" in df.columns:
            df["route_popularity"] = np.log1p(df["stats_PAS_PAS_CRD_TOTAL_y"])

        # Hub operation likelihood
        major_hubs = ["EGLL", "EDDF", "LFPG", "EHAM", "LEMD", "LTBA"]
        df["is_hub_route"] = df["adep"].isin(major_hubs) | df["ades"].isin(major_hubs)

        # Airport pair market power
        df["origin_destination_size"] = df["adep_type"].map(
            {"large_airport": 3, "medium_airport": 2, "small_airport": 1}
        ) * df["ades_type"].map(
            {"large_airport": 3, "medium_airport": 2, "small_airport": 1}
        )

    def _add_environmental_impact_features(self, df):
        """Features related to environmental considerations"""
        # Noise impact consideration
        df["noise_sensitive_departure"] = (df["departure_hour"] < 6) | (
            df["departure_hour"] > 22
        )

        # Fuel efficiency metrics
        if "cruise_fuel_flow_calculated" in df.columns:
            df["fuel_per_seat_km"] = (
                df["cruise_fuel_flow_calculated"]
                * df["flight_duration"]
                / (df["Seats Total"] * df["flown_distance"])
            )

    def _add_operational_complexity_features(self, df):
        """Features capturing operational complexity"""
        # Route complexity
        df["route_complexity"] = (
            (df["is_international"].astype(int) * 2)
            + (df["elevation_change"].abs() > 2000).astype(int)
            + ((df["flown_distance"] > 3000).astype(int) * 1.5)
            + (df["is_hub_route"].astype(int))
        )

        # Turnaround pressure
        df["tight_turnaround"] = (
            df["flight_duration"] < 90
        )  # Short flights need quick turnaround

        # Technical stop likelihood
        df["technical_stop_likely"] = (df["flown_distance"] > 0.8 * df["Range(nm)"]) & (
            df["Fuel Capacity"] < df["MTOW"] * 0.3
        )

    def _add_passenger_behavior_features(self, df):
        """Features based on passenger behavior patterns"""
        # Baggage load prediction
        df["high_baggage_likelihood"] = (
            (df["is_holiday_month"])
            | (df["flown_distance"] > 2000)
            | (df["is_international"])
        )

        # Comfort class impact
        df["comfort_index"] = (
            (df["Seats First_Class"] * 2.5)
            + (df["Seats Business_Class"] * 2.0)
            + (df["Seats Premium_Economy_Class"] * 1.5)
            + (df["Seats Economy_Class"] * 1.0)
        ) / df["Seats Total"]

        # Connection passenger likelihood
        df["connection_pax_likely"] = (
            (df["is_hub_route"])
            & (df["departure_hour"].between(8, 20))
            & (df["flown_distance"] < 2000)
        )
