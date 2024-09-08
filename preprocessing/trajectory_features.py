import pandas as pd
from pathlib import Path
from tqdm import tqdm
from functools import cache
import numpy as np
from preprocessing.base_preprocessor import BasePreprocessor
from utils.dataset import Dataset

# TODO: This needs to be a Preprocessor class
# Right now the code is way too slow... and the features dont seem to help a lot
class TrajectoryFeaturesPreprocessor(BasePreprocessor):
    def process(self, dataset: Dataset) -> Dataset:
        dataset.df = self.add_trajectory_features(dataset)
        return dataset

    @cache
    def _load_airports():
        aps = pd.read_csv("runway_data/airports.csv")
        return aps


    def _get_near_airport_traj(self,trajectory, ap):
        aps = self._load_airports()
        # find near adep
        if ap not in aps.ident.unique():
            return trajectory.head(0)

        lat, lon, alt_ft = aps[aps["ident"] == ap][
            ["latitude_deg", "longitude_deg", "elevation_ft"]
        ].values[0]
        mask = (trajectory.latitude - lat).abs() < 0.1
        mask &= (trajectory.longitude - lon).abs() < 0.1
        mask &= (trajectory.altitude - alt_ft) < 5000

        near = trajectory[mask]
        return near


    def _stats_for_series(series, name):
        res = {}
        if len(series) > 0:
            res[f"{name}_max"] = series.max()
            res[f"{name}_median"] = series.median()
            res[f"{name}_mean"] = series.mean()
            res[f"{name}_min"] = series.min()
            res[f"{name}_std"] = series.std()
        return res


    def _calculate_takeoff_features(self,trajectory):
        features = {}
        indicated_vert_rate = trajectory.vertical_rate
        measured_climb_rate = trajectory.altitude.diff() * 60
        gr_acceleration = trajectory.groundspeed.diff()  # knots / s
        gr_speed = trajectory.groundspeed  # knots

        air_speed = self._calc_air_speed(trajectory)  # knots
        air_acceleration = air_speed.diff()  # knots

        # TODO: Calculate runway length based on distance traveled and compare with runway length from airports df

        all_series = {
            "indicated_vert_rate": indicated_vert_rate,
            "measured_climb_rate": measured_climb_rate,
            "gr_acceleration": gr_acceleration,
            "gr_speed": gr_speed,
            "air_speed": air_speed,
            "air_acceleration": air_acceleration,
        }
        t_last = trajectory.timestamp.max()
        delta_t = (t_last - trajectory.timestamp).dt.total_seconds()
        for name, series in all_series.items():
            features.update(self._stats_for_series(series, f"takeoff_{name}"))
            for ds in [5, 10, 30, 60]:
                series = series[delta_t <= ds] * 60
                features.update(self._stats_for_series(series, f"takeoff_{name}_last_{ds}s"))

        return features


    def _calc_air_speed(df):
        # TODO: Written by Claude, double check if correct!
        # Convert heading to radians
        heading_rad = np.deg2rad(df["track_unwrapped"])

        # Calculate wind speed and direction
        wind_speed = np.sqrt(
            df["u_component_of_wind"] ** 2 + df["v_component_of_wind"] ** 2
        )
        wind_direction = np.arctan2(-df["u_component_of_wind"], -df["v_component_of_wind"])

        # Calculate the angle between wind direction and aircraft heading
        angle_diff = wind_direction - heading_rad

        # Calculate headwind component
        # m/s => kts
        headwind = 1.94384 * wind_speed * np.cos(angle_diff)

        # Calculate groundspeed minus headwind
        return df["groundspeed"] - headwind


    def _calculate_climb_features(self,trajectory):
        features = {}
        indicated_vert_rate = trajectory.vertical_rate
        measured_climb_rate = trajectory.altitude.diff() * 60
        gr_acceleration = trajectory.groundspeed.diff()  # knots / s
        gr_speed = trajectory.groundspeed  # knots

        air_speed = self._calc_air_speed(trajectory)  # knots
        air_acceleration = air_speed.diff()  # knots

        # TODO: Maybe look at lateral drift? Could be difficult due to wind accuracy
        # TODO: look at total time until cruising altitude
        # TODO: Max continous climb

        all_series = {
            "indicated_vert_rate": indicated_vert_rate,
            "measured_climb_rate": measured_climb_rate,
            "gr_acceleration": gr_acceleration,
            "gr_speed": gr_speed,
            "air_speed": air_speed,
            "air_acceleration": air_acceleration,
        }
        t_init = trajectory.timestamp.min()
        delta_t = (trajectory.timestamp - t_init).dt.total_seconds()
        for name, series in all_series.items():
            features.update(self._stats_for_series(series, f"climb_{name}"))
            for ds in [5, 10, 30, 60]:
                series = series[delta_t <= ds] * 60
                features.update(self._stats_for_series(series, f"climb_{name}_first_{ds}s"))

        return features


    def _calculate_traj_features(self, trajectory, flight_info):
        features = {}
        adep = flight_info["adep"]
        dep = self._get_near_airport_traj(trajectory, adep)
        if len(dep) == 0:
            return features  # TODO: for now

        measured_ground = dep.altitude.min()
        on_ground = dep[dep.altitude - measured_ground <= 200]  # 100 ft
        if len(on_ground) >= 10:
            takeoff_feat = self._calculate_takeoff_features(on_ground)
            features.update(takeoff_feat)

        climb = dep[dep.altitude - measured_ground > 200]
        if len(climb) >= 10:
            climb_rate_feat = self._calculate_climb_features(climb)
            features.update(climb_rate_feat)

        return features


    def add_trajectory_features(self, challenge):
        for date in tqdm(challenge.df.date.unique()):
            if not (Path("data") / f"{date}.parquet").exists():
                continue
            trajs = pd.read_parquet(f"data/{date}.parquet")
            trajs["flight_id"] = trajs["flight_id"].astype(int)
            flight_ids = challenge.df[challenge.df.date == date].flight_id.unique()
            for fid in tqdm(flight_ids):
                fid_info = challenge.df[challenge.df.flight_id == fid].iloc[0].to_dict()
                features = self._calculate_traj_features(trajs[trajs.flight_id == fid], fid_info)
                for feat, val in features.items():
                    challenge.df.loc[challenge.df.flight_id == fid, feat] = val
        return challenge


    # Idea:
    # 1. clean data
    # 2. filter takeoffs and landings
    # 3. find IAS (groundspeed - headwind) at start (first measurement off the ground)
    # 4. V2 speed
    # 5. Climb rates
