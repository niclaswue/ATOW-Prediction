# Aviation Dataset Column Legend

In addition to the columns provided by the `challenge_set.csv` and `submission_set.csv`, we calculated the following features with our preprocessor-pipeline:

## Flight Information
- `flight_id` - Unique identifier for each flight
- `predicted_tow` - Predicted Take-Off Weight in kilograms
- `date` - Date of the flight
- `callsign` - Unique flight identifier (encrypted)
- `aircraft_type` - Type of aircraft (e.g., A320)
- `wtc` - Wake Turbulence Category (L=Light, M=Medium, H=Heavy)
- `airline` - Operating airline (encrypted)
- `flight_duration` - Total flight time in minutes
- `taxiout_time` - Time spent taxiing before takeoff in minutes
- `flown_distance` - Actual distance flown in kilometers
- `tow` - Actual Take-Off Weight in kilograms

## Airport Information
- `adep` - Departure airport ICAO code
- `ades` - Arrival airport ICAO code
- `name_adep` - Departure airport name
- `name_ades` - Arrival airport name
- `country_code_adep` - Departure country code
- `country_code_ades` - Arrival country code
- `adep_city` - Departure city
- `ades_city` - Arrival city
- `adep_region` - Departure airport region
- `ades_region` - Arrival airport region
- `adep_type` - Departure airport size classification
- `ades_type` - Arrival airport size classification
- `adep_elevation` - Departure airport elevation in feet
- `ades_elevation_x` - Arrival airport elevation in feet

## Geographic Data
- `adep_lat` - Departure airport latitude
- `ades_lat_x` - Arrival airport latitude
- `adep_lon` - Departure airport longitude
- `ades_lon_x` - Arrival airport longitude
- `adep_tz` - Departure airport timezone
- `ades_tz` - Arrival airport timezone

## Time Information
- `actual_offblock_time` - Time aircraft leaves parking position (UTC)
- `arrival_time` - Time of arrival (UTC)
- `adep_local_offblock_time` - Local time of departure
- `ades_local_arrival_time` - Local time of arrival
- `takeoff_time` - Actual takeoff time
- `onblock_time` - Time aircraft reaches parking position
- `air_time_hours` - Time spent in the air

## Aircraft Specifications
- `Seats First_Class` - Number of first class seats
- `Seats Business_Class` - Number of business class seats
- `Seats Economy_Class` - Number of economy class seats
- `Seats Total` - Total number of seats
- `Cargo Capacity` - Maximum cargo capacity in kg
- `Range(nm)` - Maximum flight range in nautical miles
- `MLW` - Maximum Landing Weight in kg
- `MTOW` - Maximum Take-Off Weight in kg
- `ZFW` - Zero Fuel Weight (weight without fuel) in kg
- `Fuel Capacity` - Maximum fuel capacity in kg
- `Fuel Flow` - Fuel consumption rate
- `Service Ceiling(ft)` - Maximum operating altitude in feet
- `Cruising Speed(mach)` - Normal cruising speed as Mach number
- `Cost Index` - Economic factor for flight planning

## Runway Information
- `runway_ades_length_ft` - Arrival runway length in feet
- `runway_adep_length_ft` - Departure runway length in feet
- `runway_ades_he_elevation_ft` - Arrival runway higher end elevation
- `runway_ades_le_elevation_ft` - Arrival runway lower end elevation
- `runway_adep_he_elevation_ft` - Departure runway higher end elevation
- `runway_adep_le_elevation_ft` - Departure runway lower end elevation
- `runway_ades_he_displaced_threshold_ft` - Arrival runway displaced threshold at higher end
- `runway_ades_le_displaced_threshold_ft` - Arrival runway displaced threshold at lower end

## Weather Conditions (ADES)
- `ades_tmpf` - Temperature in Fahrenheit
- `ades_dwpf` - Dew point in Fahrenheit
- `ades_relh` - Relative humidity percentage
- `ades_drct` - Wind direction in degrees
- `ades_sknt` - Wind speed in knots
- `ades_alti` - Altimeter setting (atmospheric pressure)
- `ades_vsby` - Visibility in miles
- `ades_gust` - Wind gust speed
- `ades_skyc1` - Sky condition (SCT=Scattered, BKN=Broken)
- `ades_skyl1` - Sky layer height in feet
- `ades_feel` - "Feels like" temperature

## Fuel Information
- `fuel_price_adep` - Fuel price at departure airport
- `fuel_price_ades` - Fuel price at arrival airport
- Various `Quantity_Kerosene-type Jet Fuel` columns - Fuel consumption and supply statistics for airports

## Trajectory Features Overview

### Trajectory Completeness Flags
- `has_takeoff_trajectory`: Whether the trajectory includes takeoff phase
- `has_landing_trajectory`: Whether the trajectory includes landing phase
- `has_cruise_trajectory`: Whether the trajectory includes cruise phase
- `track_distance_m`: Total distance covered in meters

### Takeoff and Initial Climb Features
- `taxi_out_time_s`: Time spent taxiing before takeoff
- `takeoff_mean_acceleration`: Mean acceleration during takeoff roll
- `takeoff_max_acceleration`: Maximum acceleration during takeoff roll
- `v2_speed_kt`: Estimated V2 speed in knots (speed at positive climb rate)

### Initial Climb Performance
- `initclimb_mean_climb`: Mean vertical rate during initial climb
- `initclimb_median_climb`: Median vertical rate during initial climb
- `initclimb_max_climb`: Maximum vertical rate during initial climb
- `initclimb_mean_alt`: Mean altitude during initial climb
- `initclimb_median_alt`: Median altitude during initial climb
- `initclimb_max_alt`: Maximum altitude during initial climb

### Initial Climb Speed
- `initclimb_min_gs`: Minimum ground speed during initial climb
- `initclimb_mean_gs`: Mean ground speed during initial climb
- `initclimb_median_gs`: Median ground speed during initial climb
- `initclimb_max_gs`: Maximum ground speed during initial climb

### Cruise Features
- `cruise_altitude`: Most common cruise altitude
- `mean_cruise_speed`: Mean ground speed during cruise
- `median_cruise_speed`: Median ground speed during cruise
- `lowest_cruise_speed`: Minimum ground speed during cruise
- `highest_cruise_speed`: Maximum ground speed during cruise
- `cruise_speed_std`: Standard deviation of cruise ground speed

### Wind Data (during cruise)
- `average_headwind`: Mean headwind component
- `max_headwind`: Maximum headwind component
- `min_headwind`: Minimum headwind component
- `std_headwind`: Standard deviation of headwind component

Note: Features return -1 if the corresponding flight phase is not available in the trajectory data or if some features could not be calculated (e.g. because flight phases could not be accurately determined).
Most challenging aspect has been the incompleteness of many trajectories, but time for data cleanup was very limited. We believe there is a lot more potential to improve upon these features, since there are very rudamentary calculated at the moment.

## Statistical Data
- `stats_MOVE_ACM_TOTAL` - Total aircraft movements
- `stats_PAS_PAS_CRD_TOTAL` - Total passenger count
- `stats_T_FRM_LD_NLD_TOTAL` - Loading/unloading statistics

## Performance Metrics
- `error` - Difference between predicted and actual TOW
- `absolute_error` - Absolute value of prediction error

