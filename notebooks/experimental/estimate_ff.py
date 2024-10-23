# # pip install git+https://github.com/DGAC/Acropole.git
# import pandas as pd
# from acropole import FuelEstimator

# fe = FuelEstimator()

# flight = pd.DataFrame(
#     {
#         "typecode": ["A320", "A320", "A320", "A320"],
#         "groundspeed": [400, 410, 420, 430],
#         "altitude": [10000, 11000, 12000, 13000],
#         "vertical_rate": [2000, 1500, 1000, 500],
#         # optional features:
#         "second": [0.0, 1.0, 2.0, 3.0],
#         "airspeed": [400, 410, 420, 430],
#         "mass": [60000, 60000, 60000, 60000],
#     }
# )

# flight_fuel = fe.estimate(flight)  # flight.data if traffic flight
# from IPython import embed

# embed()
# exit()  # TODO: Remove DBG


import sys
from pathlib import Path
import pandas as pd
from traffic.core import Flight, Traffic
from multiprocessing import Pool, cpu_count
import concurrent.futures
import os
from tqdm import tqdm
import time
import json
from acropole import FuelEstimator

sys.path.append("../..")
from utils.data_loader import DataLoader


def process_flight(flight: Flight, flight_id: int):
    fe = FuelEstimator()

    flight_data = flight.data.sort_values(by="timestamp")
    flight_data = flight_data.rename(
        columns={
            "groundspeed": "groundspeed",
            "altitude": "altitude",
            "vertical_rate": "vertical_rate",
            "TAS": "airspeed",
        }
    )

    # Add required columns
    flight_data["typecode"] = (
        flight.aircraft
    )  # Assuming the aircraft type is stored in the 'aircraft' attribute
    flight_data["second"] = (
        flight_data["timestamp"] - flight_data["timestamp"].min()
    ).dt.total_seconds()

    # Estimate mass (you may need to implement a more accurate method)
    flight_data["mass"] = (
        60000  # Example static mass, replace with dynamic calculation if possible
    )

    try:
        flight_fuel = fe.estimate(flight_data)

        fuel_info = {
            "flight_id": flight_id,
            "total_fuel_consumption": float(flight_fuel["fuel"].sum()),
            "avg_fuel_consumption": float(flight_fuel["fuel"].mean()),
            "max_fuel_consumption": float(flight_fuel["fuel"].max()),
            "flight_duration": float(flight_data["second"].max()),
            "info": "OK",
        }
    except Exception as e:
        fuel_info = {"flight_id": flight_id, "info": f"Error: {str(e)}"}

    return fuel_info


def process_date(date):
    ld = DataLoader(Path("/home/wues_ni/Projects/ATOW-Prediction/data"))
    ds, _, _ = ld.load()
    t = Traffic.from_file(f"/home/wues_ni/Projects/ATOW-Prediction/data/{date}.parquet")

    flight_ids = t.data.flight_id.unique()

    preprocessed = []
    for flight_id in flight_ids:
        f = t.query(f"flight_id == {flight_id}")
        if len(f) > 0:
            preprocessed.append((f[0], int(flight_id)))

    result = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
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

    results = []
    for date in tqdm(dates, desc="Processing dates"):
        results.extend(process_date(date))

    print(f"Processed {len(results)} flights")

    with open("fuel_consumption_info.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
