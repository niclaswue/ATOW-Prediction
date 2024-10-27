# PRC Data Challenge Submission (team_faithful_engine)
This is our entry for the PRC Data Challenge: https://ansperformance.eu/study/data-challenge
We are committed to go open on the outcome of this challenge!

>[!Note]
> This README focusses on setting up and running the project. A more [general overview](documentation/project_overview.md) of the project is also available.


## Initial Setup
### Setup up the development environment

Create a new conda environment with Python 3.11
```
conda create -n tow -c conda-forge python=3.11 -y
conda activate tow
```

Note: For future trajectory analysis and feature creation, you may want to create another separate environment around traffic.
See: https://traffic-viz.github.io/installation.html

Install all the required packages:
```
pip install -r requirements.txt
```
### Downloading all datasets

Now it's time to download the competition data.
This script will create a new directory called `data` and download the competition data.
It will start with the mandatory csv files and then continues with the daily trajectories.
> [!NOTE]
> These are 150GB+ in size, so downloading will take a while. You might want to start the script in a screen session.
> You can stop the download anytime, it will automatically resume when started again. Alternatively, you can manually copy relevant OSN Trajectory `.parquet` files into `data/`.


```
python download_competition_data.py
```

Next we download the additional datasets that were used to boost the performance. All used datasets, licenses and attribution can be found under the Dataset Overview chapter.

```
python download_additional_data.py
```
This will download the simple to fetch tabular datasets to the `additional_data` directory.

We also use daily METAR weather data. Right now, we only download the METARs for the destination airports.
The following script will gather all unique destination airports for each day and download the reports for them.
In the end, they are combined to one large weather dataset. All downloaded links are saved in `processed.txt` which allows you to resume the download if needed.
```
python download_weather_data.py
```

Additionally, in the future, we might use T-100 forms data from the bureau of transportation statistics.
This data can be downloaded by hand, however for your conveinence we provide a scraper. To not overload the poor server we wait a long time in between requests, therefore the download process will take a long time. 

```
python download_bts_t100.py.py
```
The resulting data is not used right now.

For an overview of all the additional datasets see the list of [additional data sources](documentation/additional_data_sources.md).

### Prepare Trajectory Features
Our model takes some input features from the OSN Trajectories. Running the Preprocessing of the Trajectories can take a while, therefore this is done in a separate step and the result is saved as `all_trajectory_features.parquet` in the `additional_data` directory.
Excpect this to take multiple hours (up to 10 hours on a regular Laptop PC).
```
python ./preprocessing/trajectory_batchprocessing.py
```
> [!TIP]
> If you have a more performant machine, edit the number of parallel processes in the constant `POOL_NUMBER` in `preprocessing/trajectory_batchprocessing.py`

Once all data is downloaded and the trajectory-features are preprocessed, you can continue with running the training.

### Run the training

To run the training, start:
```
python run.py
```

### Logging with Weights & Biases
Create a free personal account at wandb.ai, then after pip installing wandb log in using `wand login`.
Afterwards, you can use the the wandb training:

```python
python run_wandb.py
```


TODO: Provide download links for the preprocessed datasets, otherwise it takes a long time to preprocess the data 

# Structure of the repository

The repository is organized as follows:

```python
├── data/                           # Directory for storing downloaded competition data
├── additional_data/                # Directory for storing additional datasets
├── scripts/                        # Directory for various utility scripts
│   ├── download_competition_data.py  # Script to download competition data
│   ├── download_additional_data.py   # Script to download additional datasets
│   ├── download_weather_data.py      # Script to download METAR weather data
│   ├── download_bts_t100.py          # Script to scrape T-100 forms data
│   └── run.py                        # Script to run the training process
├── requirements.txt                # List of required Python packages
├── models                          # Directory for storing different AI models used in training
├── README.md                       # Project overview and setup instructions
├── documentation/                  # Directory for project documentation
│   └── additional_data_sources.md    # Documentation for additional data sources
└── museum/                         # Collection of scripts and notebooks we used during development, not relevant for data pipeline
```

### Key Modules and Classes

- **download_competition_data.py**: Handles downloading of the main competition data, including the OSN trajectory files.
- **download_additional_data.py**: Manages downloading of supplementary datasets to enhance model performance.
- **download_weather_data.py**: Gathers METAR weather data for destination airports and compiles it into a comprehensive dataset.
- **download_bts_t100.py**: Scrapes T-100 forms data from the Bureau of Transportation Statistics, though this data is not currently used.
- **run.py**: Main script to initiate the training process.
- **run_wandb.py**: Main script to initiate the training process, with model information stored in [Weights&Biases](https://wandb.ai) for MLOps.
- **preprocessing directory**: This directory contains the preprocessors used to extract features from the various datasets.

### Additional Documentation

- [**additional_data_sources.md**](documentation/additional_data_sources.md): Lists all additional datasets used, along with their licenses and attributions.
- [**project_overview.md**](documentation/project_overview.md): General project overview

### Data Directories

- **data/**: Contains the primary competition data.
- **additional_data/**: Stores additional datasets that are used to improve model performance.


# Aviation Dataset Column Legend

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

## Statistical Data
- `stats_MOVE_ACM_TOTAL` - Total aircraft movements
- `stats_PAS_PAS_CRD_TOTAL` - Total passenger count
- `stats_T_FRM_LD_NLD_TOTAL` - Loading/unloading statistics

## Performance Metrics
- `error` - Difference between predicted and actual TOW
- `absolute_error` - Absolute value of prediction error

