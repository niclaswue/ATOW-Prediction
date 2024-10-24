#
# here i try to do the calculations for each trajecotry once and save the result, because the preprocessing takes a lot of time
#

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from traffic.core import Flight

root_dir = Path(__file__).parent.parent.absolute()
additional_data_dir = root_dir / "additional_data"
trajectory_data_dir = root_dir / "data"
single_flight_data_dir = additional_data_dir / "single_flight_data"
flight_information_file = trajectory_data_dir / "challenge_set.csv"

SPEED_THRESHOLD = 35 #knots
POOL_NUMBER = 1 # choose 1 for no parallel processing

flight_information = None

def main() -> None:
    create_trajectory_features_batch()
    
def create_trajectory_features_batch() -> None:
    try:
        flight_information = pd.read_csv(flight_information_file)
    except FileNotFoundError:
        print("TrajectoryPreprocessor: Flight information file not found.")
        return
    
    #split_trajectories_into_single_flights()
    for date_file in tqdm(list(trajectory_data_dir.glob("*.parquet"))):
        date = date_file.stem
        date_df = pd.read_parquet(date_file)
        
        with Pool(POOL_NUMBER) as p:
            result = p.starmap(create_trajectory_features, zip([flight_id for flight_id in date_df["flight_id"].unique()], repeat(date_df)) )
            result_df = pd.concat(result, ignore_index=True)
            result_df.to_parquet(additional_data_dir / "trajectory_features" / f"{date}.parquet", index=False)
        
    # combine all the parquet files into one
    all_files = list(additional_data_dir.glob("trajectory_features/\d{4}-\d{2}-\d{2}\.parquet"))
    all_data = pd.concat([pd.read_parquet(file) for file in all_files])
    all_data.to_parquet(additional_data_dir / "all_trajectory_features.parquet", index=False)    
    
         
            
def create_trajectory_features(flight_id, trajectory) -> pd.DataFrame:
    flight = Flight(trajectory)
    flight = flight.phases(twindow=60)
    flight = estimate_fuel_flow(flight)
    flight = flight.compute_distance()
    flight = flight.compute_TAS()
    
    
    result = {}
    result["flight_id"] = flight_id
    
    #track_distance_m = calculate_track_distance_m(trajectory) # should be the same as using flight.compute_distance()
    track_distance_m = flight.data["cumdist"].iloc[-1]
    
    result["track_distance_m"] = track_distance_m
    
    # not all trajectories are complete, so we check if the trajectory has a takeoff, landing and cruise phase
    result["has_takeoff_trajectory"] = has_takeoff_trajectory(trajectory)
    result["has_landing_trajectory"] = has_landing_trajectory(trajectory)
    result["has_cruise_trajectory"] = has_cruise_trajectory(trajectory)
   
   
    result["fuel_burnt_kg"] = flight.data["fuel"].iloc[-1]
    
    if result["has_takeoff_trajectory"]:
        result.update(calculate_takeoff_features(flight))
        
    else:
        # use negative values for the features if the takeoff trajectory is not complete
        result["taxi_out_time_s"] = -1
        result["takeoff_mean_acceleration"] = -1
        result["takeoff_max_acceleration"] = -1
        result["takeoff_roll_distance_m"] = -1
        result["v2_speed_kt"] = -1
        result["initclimb_mean_climb"] = -1
        result["initclimb_median_climb"] = -1
        result["initclimb_max_climb"] = -1
        result["initclimb_mean_alt"] = -1
        result["initclimb_median_alt"] = -1
        result["initclimb_max_alt"] = -1
        result["initclimb_min_tas"] = -1
        result["initclimb_mean_tas"] = -1
        result["initclimb_median_tas"] = -1
        result["initclimb_max_tas"] = -1
        result["initclimb_min_gs"] = -1
        result["initclimb_mean_gs"] = -1
        result["initclimb_median_gs"] = -1
        result["initclimb_max_gs"] = -1
        
    
    return pd.DataFrame([result])
    
    
def calculate_takeoff_features(flight: Flight) -> dict:
    result = {}
    climb = get_initial_climb_trajectory(flight)
    
    taxi_out_time_s = calculate_taxi_out_time(climb)
    result["taxi_out_time_s"] = taxi_out_time_s

    acceleration = calculate_acceleration_on_takeoff_run(climb)
    result["takeoff_mean_acceleration"] = acceleration.mean()
    result["takeoff_max_acceleration"] = acceleration.max()

    takeoff_roll_distance_m = calculate_takeoff_roll_distance_m(climb)
    result["takeoff_roll_distance_m"] = takeoff_roll_distance_m

    v2_speed = get_v2_speed(climb)
    result["v2_speed_kt"] = v2_speed
    
    result["initclimb_mean_climb"] = climb["vertical_rate"].mean()
    result["initclimb_median_climb"] = climb["vertical_rate"].median()
    result["initclimb_max_climb"] = climb["vertical_rate"].max()
    result["initclimb_mean_alt"] = climb["altitude"].mean()
    result["initclimb_median_alt"] = climb["altitude"].median()
    result["initclimb_max_alt"] = climb["altitude"].max()
    result["initclimb_min_tas"] = climb["TAS"].min()
    result["initclimb_mean_tas"] = climb["TAS"].mean()
    result["initclimb_median_tas"] = climb["TAS"].median()
    result["initclimb_max_tas"] = climb["TAS"].max()
    result["initclimb_min_gs"] = climb["groundspeed"].min()
    result["initclimb_mean_gs"] = climb["groundspeed"].mean()
    result["initclimb_median_gs"] = climb["groundspeed"].median()
    result["initclimb_max_gs"] = climb["groundspeed"].max()
    
    return result

def has_takeoff_trajectory(flight: Flight) -> bool:
    adep = flight_information[flight_information["flight_id"] == flight.flight_id]["adep"].values[0]
    return flight.takeoff_from(adep)

def has_landing_trajectory(flight: Flight) -> bool:
    ades = flight_information[flight_information["flight_id"] == flight.flight_id]["ades"].values[0]
    return flight.landing_at(ades)


def has_cruise_trajectory(flight: Flight) -> bool:
    return not flight.data.query('10000 < altitude < 40000').empty
    
    
def split_trajectories_into_single_flights() -> None:
    if not single_flight_data_dir.exists():
        single_flight_data_dir.mkdir()
    if not any(single_flight_data_dir.iterdir()):
        for date_file in tqdm(list(trajectory_data_dir.glob("*.parquet"))):
            date = date_file.stem
            date_df = pd.read_parquet(date_file)
            for flight_id in date_df["flight_id"].unique():
                create_single_flight_parquet(flight_id, date_df)
          
def create_single_flight_parquet(flight_id, trajectories) -> None:
    trajectory = trajectories[trajectories["flight_id"] == flight_id]
    trajectory.to_parquet(single_flight_data_dir / f"{flight_id}.parquet", index=False)
    
    
def get_initial_climb_trajectory( flight: Flight) -> Flight:
    climbs = flight.data[flight.data["phase"] == "CLIMB"]
    # get the last index of the first climb phase. we have to delete the first index because else it is always 0
    last_index_of_first_climb = climbs.index.to_series.diff().ne(1).iloc[1:].idxmax() 
    climb_trajectory = flight.data[:last_index_of_first_climb]
        
    # Old code from the original preprocessing script, obsolete with the traffic library
    # simple function to get the initial climb trajectory of the aircraft until it reaches an altitude of 5000 ft
    #first_index_above_5000 = trajectory[trajectory["altitude"] > 5000].index[0] if not trajectory[trajectory["altitude"] > 5000].empty else len(trajectory)
    #climb_trajectory = trajectory[:first_index_above_5000]
    return climb_trajectory

def calculate_acceleration_on_takeoff_run(flight: Flight) -> pd.Series:
    try:
        #calculate acceleration in m/s^2, while considering the time difference between each point in the trajectory
        first_index_above_35knots = flight.data[flight.data["groundspeed"] > SPEED_THRESHOLD ].index[0]
        
        # i look at the first minute of the takeoff to calculate the acceleration
        takeoff_and_init_climb = flight.data[first_index_above_35knots-20:first_index_above_35knots+40]

        #calculate acceleration to previous trajectory point
        acceleration = (takeoff_and_init_climb["groundspeed"].diff() / (takeoff_and_init_climb["timestamp"].diff().dt.total_seconds())) * 0.514444 # convert from knots/s to m/s^2
        
        return acceleration
    except:
        return pd.Series([0])

def calculate_track_distance_m(trajectory: pd.DataFrame) -> float:
    # TODO check if this differs from the distance in the dataset
    # might not be necessary to calculate this, as the distance is already in the dataset -> traffic library
    # Calculate the total distance covered by the aircraft in the trajectory using haversine formula
    lat1 = np.radians(trajectory["latitude"][:-1].values)
    lon1 = np.radians(trajectory["longitude"][:-1].values)
    lat2 = np.radians(trajectory["latitude"][1:].values)
    lon2 = np.radians(trajectory["longitude"][1:].values)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    R = 6371000  # radius of the earth in meters
    distances = R * c
    
    return distances.sum()
    
def calculate_taxi_out_time(flight: Flight) -> float:
    try:
        # Calculate the time taken for the aircraft to taxi out from the gate to the runway
        # Taxi out time is the time taken from the first point in the trajectory where the speed exceeds 2 knots to the first point where it exceeds 35 knots
        # TODO check whether this matches the values in the dataset
        # TODO check if the speed below 35 knots is a good cutoff point (would not excpect any plane to taxi faster than this...)
        
        taxi_out_traj = flight.data[(flight.data["groundspeed"] > 2) & (flight.ata["groundspeed"] < SPEED_THRESHOLD)]
        taxi_out_time = (taxi_out_traj["timestamp"].max() - taxi_out_traj["timestamp"].min())
        return taxi_out_time.total_seconds()
    except:
        return 0
    
def calculate_takeoff_roll_distance_m(flight: Flight) -> float:
    try:
        # look at the first time speed is above 35 knots, 
        first_index_above_35knots = find_first_index_with_streak_above(flight, "groundspeed", SPEED_THRESHOLD, count=5)
        # from there, go back to the lowest previous speed before it rises again to set the point where the takeoff roll starts
        takeoff_roll_start = first_index_above_35knots
        for i in range(first_index_above_35knots, 0, -1):
            if flight.data.loc[i, "groundspeed"] < flight.data.loc[takeoff_roll_start, "groundspeed"]:
                takeoff_roll_start = i
            if flight.data.loc[i, "groundspeed"] > flight.data.loc[takeoff_roll_start, "groundspeed"]:
                # speed increases further back, so we break the loop
                break
        
        # takeoff roll ends when vertical speed is positive (say above 2 m/s)	
        takeoff_roll_end = flight.data[flight.data["vertical_rate"] > 100].index[0]
        
        # calculate haversine distance between the two points
        lat1 = np.radians(flight.data.loc[takeoff_roll_start, "latitude"])
        lon1 = np.radians(flight.data.loc[takeoff_roll_start, "longitude"])
        lat2 = np.radians(flight.data.loc[takeoff_roll_end, "latitude"])
        lon2 = np.radians(flight.data.loc[takeoff_roll_end, "longitude"])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        R = 6371000  # radius of the earth in meters
        distance = R * c
        
        return distance
    except:
        return 0

def get_v2_speed(flight: Flight) -> float:
    try:
        # very basic estimation of the v2 speed
        first_index_positive_climb_rate = flight.data[flight.data["vertical_rate"] > 100].index[0]
        v2 = flight.data.loc[first_index_positive_climb_rate, "groundspeed"]
        return v2
    except:
        return 0
    
def get_cruise_data(flight: Flight) -> dict:
    # get the longest level flight segment and the altitude of the aircraft during this segment with mean cruise speed
    #begin_of_phase = flight["phase"].ne(flight["phase"].shift())
    #segments_ids = begin_of_phase.cumsum()
    
    cruise_altitude = flight.data["altitude"].value_counts().idxmax()
    
    # Idea: iterate through the flight and find the longest consecutive streak of the cruise altitude to get the longest cruise phase for altitude and speed calculation
    # 
    for i, row in flight.data.iterrows():
        if row['altitude'] == cruise_altitude:
            if current_streak == 0:
                current_start_index = i
            current_streak += 1
        else:
            if current_streak > longest_streak:
                longest_streak = current_streak
                start_index = current_start_index
                end_index = i - 1
            current_streak = 0

    # Check the last streak
    if current_streak > longest_streak:
        longest_streak = current_streak
        start_index = current_start_index
        end_index = len(flight.data) - 1

    longest_cruise_phase = flight.data.iloc[start_index:end_index]
    mean_cruise_speed = longest_cruise_phase["groundspeed"].mean()
    longest_cruise_duration = longest_cruise_phase["timestamp"].max() - longest_cruise_phase["timestamp"].min() 

    return {
        'altitude': cruise_altitude,
        'longest_cruise_duration': longest_cruise_duration.total_seconds(),
        'mean_cruise_speed': mean_cruise_speed
    }
    
def find_first_index_with_streak_above(data, column, value, count=10) -> int:
    # this function returns the first index, where a streak begins with at least count values above the given value, or last index if no such streak is found
    streak = 0
    for i, row in data.iterrows():
        if row[column] > value:
            streak += 1
            if streak >= count:
                # streak of at least count values above the threshold
                # return the index where the streak started
                return i - count + 1
        else:
            streak = 0
    return len(data) - 1
        
def has_diverted(flight: Flight) -> bool:
    # check if the aircraft has diverted from its original destination
    # TODO we need to have a destination in the data to use this function
    return flight.data["diverted"].any()
    
def estimate_fuel_flow(flight: Flight) -> Flight:
    # estimate the fuel flow of the aircraft during the flight by using the tow from the competition data
    initial_mass = flight_information[flight_information["flight_id"] == flight.flight_id]["tow"].values[0]
    typecode = flight_information[flight_information["flight_id"] == flight.flight_id]["aircraft_type"].values[0]
    
    # TODO: we could find out the engine type most probably used by the airline and actype maybe? 
    
    flight = flight.fuelflow(initial_mass=initial_mass, typecode=typecode)
    return flight
    
if __name__ == "__main__":
    main()