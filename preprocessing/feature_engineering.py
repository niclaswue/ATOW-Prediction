from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset


class FeatureEngineeringPreprocessor(BasePreprocessor):
    def __init__(self):
        # Constants for calculations
        self.kg_to_lbs = 2.20462
        self.ft_to_meters = 0.3048
        self.nm_to_km = 1.852
        self.std_pax_weight = 100  # kg with baggage

    def process(self, dataset: Dataset) -> Dataset:
        print("Adding optimized takeoff weight prediction features...")

        df = dataset.df

        # High importance feature groups
        self._add_aircraft_characteristics(df)
        self._add_route_features(df)
        self._add_weight_limit_features(df)
        self._add_performance_features(df)
        self._add_payload_features(df)

        print("Done.")
        return dataset

    def _add_aircraft_characteristics(self, df):
        """Aircraft type and characteristics (highest importance features)"""
        # Create WTC-related features (highest importance)
        df["is_heavy"] = df["wtc"] == "H"
        df["is_medium"] = df["wtc"] == "M"
        df["is_light"] = df["wtc"] == "L"

        # Engine configuration (part of aircraft characteristics)
        if "openap_engine.number" in df.columns:
            df["total_thrust_capacity"] = (
                df["openap_max_thrust"] * df["openap_engine.number"]
            )
            df["thrust_per_engine"] = df["openap_max_thrust"]

        # Wing characteristics
        if all(col in df.columns for col in ["openap_wing.area", "openap_wing.span"]):
            df["wing_aspect_ratio"] = (df["openap_wing.span"] ** 2) / df[
                "openap_wing.area"
            ]
            df["wing_loading_capacity"] = df["MTOW"] / df["openap_wing.area"]

        # Aircraft size metrics
        df["fuselage_volume"] = (
            df["openap_fuselage.length"]
            * df["openap_fuselage.height"]
            * df["openap_fuselage.width"]
        )

        # Engine characteristics
        df["engine_bypass_ratio"] = df["openap_bpr"]
        df["engine_pressure_ratio"] = df["openap_pr"]

    def _add_route_features(self, df):
        """Route and distance features (high importance)"""
        # Basic route characteristics
        df["route_type"] = df.apply(
            lambda x: self._classify_route(
                x["flown_distance"],
                x["is_international"] if "is_international" in df.columns else False,
            ),
            axis=1,
        )

        # Distance ratios (high importance features)
        df["distance_to_range_ratio"] = df["flown_distance"] / df["Range(nm)"]
        df["distance_vs_ceiling"] = df["flown_distance"] / df["Service Ceiling(ft)"]

        # Elevation impacts
        df["elevation_change"] = df["adep_elevation"] - df["ades_elevation_x"]
        df["elevation_factor"] = 1 - (
            df["adep_elevation"] / 145000
        )  # Atmospheric pressure factor

        # Route efficiency
        if "track_distance_m" in df.columns and "route_distance_km" in df.columns:
            df["route_efficiency"] = df["track_distance_m"] / df["route_distance_km"]

        # Flight time features
        if "flight_duration" in df.columns:
            df["avg_speed"] = df["flown_distance"] / df["flight_duration"]
            df["flight_time_vs_optimal"] = df["flight_duration"] / (
                df["flown_distance"] / 450
            )  # 450 knots typical cruise

    def _add_weight_limit_features(self, df):
        """Weight limits and related features (high importance)"""
        # Structural weight ratios
        df["mlw_to_mtow_ratio"] = df["MLW"] / df["MTOW"]
        df["zfw_to_mtow_ratio"] = df["ZFW"] / df["MTOW"]
        df["empty_to_mtow_ratio"] = df["openap_oew"] / df["MTOW"]

        # Available weight capacities
        df["max_payload_structural"] = df["ZFW"] - df["openap_oew"]
        df["max_fuel_structural"] = df["MTOW"] - df["ZFW"]

        # Fuel capacity constraints
        df["fuel_capacity_ratio"] = df["Fuel Capacity"] / df["MTOW"]
        df["fuel_range_factor"] = df["Fuel Capacity"] / (
            df["flown_distance"] + 100
        )  # +100 for reserve

        # Operating weight ratios
        df["operating_empty_ratio"] = df["openap_oew"] / df["MTOW"]
        df["useful_load_ratio"] = (df["MTOW"] - df["openap_oew"]) / df["MTOW"]

    def _add_performance_features(self, df):
        """Performance-related features (high importance)"""
        # Cruise performance (high importance)
        df["altitude_ratio"] = df["cruise_altitude"] / df["Service Ceiling(ft)"]

        # Speed characteristics
        if "mean_cruise_speed" in df.columns:
            df["speed_efficiency"] = df["mean_cruise_speed"] / df["openap_vmo"]
            df["speed_consistency"] = 1 - (
                df["cruise_speed_std"] / df["mean_cruise_speed"]
            )

        # Wind impact (moderately high importance)
        if "average_headwind" in df.columns:
            df["headwind_factor"] = df["average_headwind"] / df["mean_cruise_speed"]
            df["wind_variation"] = (df["max_headwind"] - df["min_headwind"]) / abs(
                df["average_headwind"]
            )

        # Performance at takeoff
        if "v2_speed_kt" in df.columns:
            df["v2_ratio"] = df["v2_speed_kt"] / df["openap_vmo"]

    def _add_payload_features(self, df):
        """Payload-related features"""
        # Passenger configuration impact
        df["total_seat_ratio"] = df["Seats Total"] / df["openap_pax.max"]
        df["premium_seat_ratio"] = (
            df["Seats First_Class"] + df["Seats Business_Class"]
        ) / df["Seats Total"]

        # Estimated weights
        df["estimated_pax_weight"] = df["Seats Total"] * self.std_pax_weight
        df["available_cargo_weight"] = (
            df["max_payload_structural"] - df["estimated_pax_weight"]
        )

        # Cargo capacity utilization potential
        if "Cargo Capacity" in df.columns:
            df["cargo_capacity_ratio"] = (
                df["available_cargo_weight"] / df["Cargo Capacity"]
            )

    @staticmethod
    def _classify_route(distance, is_international):
        """Helper to classify route types"""
        if distance <= 500:
            return 1  # Short haul
        elif distance <= 3000:
            return 2  # Medium haul
        else:
            return 3  # Long haul
