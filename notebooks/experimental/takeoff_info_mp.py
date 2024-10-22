import sys
from pathlib import Path
import pandas as pd
from traffic.core import Flight, Traffic
import matplotlib.pyplot as plt
from geopy import distance
from multiprocessing import Pool, cpu_count
import concurrent.futures
import os
from tqdm import tqdm
import time
import json
from traffic.data import airports

sys.path.append("../..")
from utils.data_loader import DataLoader


def process_flight(f: Flight, adep: str, flight_id: int, runways: pd.DataFrame):
    # print(flight_id)
    f.data.sort_values(by="timestamp", inplace=True)
    f = f.filter(altitude=(17, 53))
    f = f.compute_TAS()
    try:
        takeoff_ap = f.takeoff_airport()
        ap = airports[adep]
        if takeoff_ap.icao != adep:
            print("takeoff_ap.icao != adep", takeoff_ap.icao, adep)
            return {
                "flight_id": flight_id,
                "info": f"Wrong airport? {takeoff_ap.icao}, {adep}",
            }
        to = f.takeoff_from_runway(ap)
    except ValueError:
        return {"flight_id": flight_id, "info": "Airport not in dataset"}

    if len(to) == 0:
        return {"flight_id": flight_id, "info": "No takeoff"}

    to = to[0]
    runway_altitude = to.data.altitude.min()
    takeoff = to.data[to.data["altitude"] >= runway_altitude]
    if "LEVEL" not in takeoff.phase.unique() or "CLIMB" not in takeoff.phase.unique():
        return {"flight_id": flight_id, "info": "Only climb or level phase."}

    acceleration = takeoff[
        # (takeoff["vertical_rate"] <= 100)
        # & (takeoff["altitude"] == runway_altitude)
        (takeoff["phase"] == "LEVEL")
    ]
    if len(acceleration) == 0:
        return {"flight_id": flight_id, "info": "No acceleration"}
    end_of_accel = acceleration.iloc[-1]
    rwy = end_of_accel["runway"]

    if rwy in runways["he_ident"].unique():
        oa_info = runways[runways["he_ident"] == rwy]
        end = oa_info[["he_latitude_deg", "he_longitude_deg"]].values
        length = oa_info["length_ft"].values
    elif rwy in runways["le_ident"].unique():
        oa_info = runways[runways["le_ident"] == rwy]
        end = oa_info[["le_latitude_deg", "le_longitude_deg"]].values
        length = oa_info["length_ft"].values
    else:
        return {"flight_id": flight_id, "info": "No runway"}

    pos = end_of_accel[["latitude", "longitude"]].values.astype(float)
    runway_left = distance.distance(pos, end).ft
    runway_used = length - runway_left
    runway_percent_used = runway_used / length

    climb = to.data[to.data["phase"] == "CLIMB"]
    altitude_climb_rate = climb["altitude"].diff()
    last_climb_pos = climb.iloc[-1][["latitude", "longitude"]].values.astype(float)

    climb_features = {
        "mean_climb": climb["vertical_rate"].mean(),
        "median_climb": climb["vertical_rate"].median(),
        "max_climb": climb["vertical_rate"].max(),
        "mean_alt": climb["altitude"].mean(),
        "median_alt": climb["altitude"].median(),
        "max_alt": climb["altitude"].max(),
        "min_tas": climb["TAS"].min(),
        "mean_tas": climb["TAS"].mean(),
        "median_tas": climb["TAS"].median(),
        "max_tas": climb["TAS"].max(),
        "min_gs": climb["groundspeed"].min(),
        "mean_gs": climb["groundspeed"].mean(),
        "median_gs": climb["groundspeed"].median(),
        "max_gs": climb["groundspeed"].max(),
        "mean_alt_climb": altitude_climb_rate.mean(),
        "median_alt_climb": altitude_climb_rate.median(),
        "max_alt_climb": altitude_climb_rate.max(),
        "total_altitude_climbed": altitude_climb_rate.sum(),
        "dist_to_rwy_end": float(distance.distance(last_climb_pos, end).ft),
        "dist_to_takeoff_point": float(distance.distance(last_climb_pos, pos).ft),
    }

    features = {
        "flight_id": flight_id,
        "runway": rwy,
        "takeoff_TAS": float(end_of_accel["TAS"]),
        "takeoff_groundspeed": float(end_of_accel["groundspeed"]),
        "takeoff_cumdist": float(
            end_of_accel["cumdist"] - acceleration.iloc[0]["cumdist"]
        ),
        "runway_ft_left": float(runway_left),
        "runway_used": float(runway_used[0]),
        "runway_percent_used": float(runway_percent_used[0]),
        "climb": climb_features,
        "info": "OK",
    }

    return features


def process_flight_climb(f: Flight, adep: str, flight_id: int, runways: pd.DataFrame):
    # print(flight_id)
    f.data.sort_values(by="timestamp", inplace=True)
    f = f.filter(altitude=(17, 53))
    f = f.compute_TAS()
    try:
        takeoff_ap = f.takeoff_airport()
        to = f.takeoff_from_runway(takeoff_ap)
    except ValueError:
        return {"flight_id": flight_id, "info": "Airport not in dataset"}

    if len(to) == 0:
        return {"flight_id": flight_id, "info": "No takeoff"}

    takeoff = to[0]
    runway_altitude = to.data.altitude.mode().values[0]
    climb = to.data[to.data["altitude"] > runway_altitude]
    if "LEVEL" not in takeoff.phase.unique() or "CLIMB" not in takeoff.phase.unique():
        return {"flight_id": flight_id, "info": "Only climb or level phase."}

    acceleration = takeoff[
        # (takeoff["vertical_rate"] <= 64)
        # & (takeoff["altitude"] == runway_altitude)
        (takeoff["phase"] == "LEVEL")
    ]
    if len(acceleration) == 0:
        return {"flight_id": flight_id, "info": "No acceleration"}

    end_of_accel = acceleration.iloc[-1]
    rwy = end_of_accel["runway"]

    if rwy in runways["he_ident"].unique():
        oa_info = runways[runways["he_ident"] == rwy]
        end = oa_info[["he_latitude_deg", "he_longitude_deg"]].values
        length = oa_info["length_ft"].values
    elif rwy in runways["le_ident"].unique():
        oa_info = runways[runways["le_ident"] == rwy]
        end = oa_info[["le_latitude_deg", "le_longitude_deg"]].values
        length = oa_info["length_ft"].values
    else:
        return {"flight_id": flight_id, "info": "No runway"}

    pos = end_of_accel[["latitude", "longitude"]].values.astype(float)
    runway_left = distance.distance(pos, end).ft
    runway_used = length - runway_left
    runway_percent_used = runway_used / length

    features = {
        "flight_id": flight_id,
        "runway": rwy,
        "takeoff_TAS": float(end_of_accel["TAS"]),
        "takeoff_groundspeed": float(end_of_accel["groundspeed"]),
        "takeoff_cumdist": float(
            end_of_accel["cumdist"] - acceleration.iloc[0]["cumdist"]
        ),
        "runway_ft_left": float(runway_left),
        "runway_used": float(runway_used[0]),
        "runway_percent_used": float(runway_percent_used[0]),
        "info": "OK",
    }

    return features


def get_data(flight_id, ds, runways, t):
    flight_info = ds.df[ds.df["flight_id"] == flight_id]
    if len(flight_info) == 0:
        return None

    adep = flight_info["adep"].values[0]
    runway_info = runways[runways["airport_ident"] == adep]
    f = t.query(f"flight_id == {flight_id}")[0]

    return (f, adep, flight_id, runway_info)


def process_date(date):
    ld = DataLoader(Path("/home/wues_ni/Projects/ATOW-Prediction/data"))
    ds, _, _ = ld.load()
    runways = pd.read_csv(
        "/home/wues_ni/Projects/ATOW-Prediction/additional_data/runway_data/runways.csv"
    )
    t = Traffic.from_file(f"/home/wues_ni/Projects/ATOW-Prediction/data/{date}.parquet")

    flight_ids = t.data.flight_id.unique()

    t.data.rename(
        {"u_component_of_wind": "wind_u", "v_component_of_wind": "wind_v"},
        axis=1,
        inplace=True,
    )

    preprocessed = []
    for flight_id in tqdm(flight_ids):
        flight_info = ds.df[ds.df["flight_id"] == flight_id]
        if len(flight_info) == 0:
            continue
        adep = flight_info["adep"].values[0]
        runway_info = runways[runways["airport_ident"] == adep]
        f = t.query(f"flight_id == {flight_id}")[0]
        preprocessed.append((f, adep, int(flight_id), runway_info))

    result = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=80) as executor:
        futures = [
            executor.submit(process_flight, *features) for features in preprocessed
        ]
        for future in concurrent.futures.as_completed(futures):
            result.append(future.result())

    print(f"Done with date {date}")
    return result


def main():
    ld = DataLoader(Path("/home/wues_ni/Projects/ATOW-Prediction/data"))
    ds, _, _ = ld.load()
    dates = ds.df["date"].unique()

    # Use all available CPU cores, or limit to a specific number if needed
    results = [process_date(date) for date in dates]

    # Process or save the final_result as needed
    print(f"Processed {len(results)} dates")

    with open("takeoff_info.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
