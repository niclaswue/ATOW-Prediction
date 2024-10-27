# PRC Data Challenge Submission (team_faithful_engine)
This is our entry for the PRC Data Challenge: https://ansperformance.eu/study/data-challenge
We are committed to go open on the outcome of this challenge!

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
These are 150GB+ in size, you might want to start the script in a screen session.
You can stop the download anytime, it will automatically resume when started again.

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
└── notebooks/                      # Jupyter notebooks for exploratory data analysis and experiments
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

- **additional_data_sources.md**: Lists all additional datasets used, along with their licenses and attributions.

### Data Directories

- **data/**: Contains the primary competition data.
- **additional_data/**: Stores additional datasets that are used to improve model performance.

### Notebooks

- **notebooks/**: Includes Jupyter notebooks for data exploration, feature engineering, and model experimentation.


---


# TODOs:
Add features to improve the prediction.
All multiplications and divisions should be features.

Feature ideas:
- zero fuel weight
- estimated engine type by knowing (Austrian, Swiss, Vueling)
- take off speed
- take off climb rate
- taxi out fuel consumption
- landing speed
- weather at destination
- weather at departure airport
- estimated fuel burn for route (need to account for flight phases)
- airline specific differences => cluster similar data per airline
- route schedule (daily, weekly etc.)
- month of year?

further improvements
- feature importance
- normalize features
- bayesian hyperparameter search
- use neural network as predictor (learn JAX?) 
- incorporate uncertainty?!

k-fold cross validation to get a good signal without waiting for the leaderboard

clean data remove quasi duplicates with same tow

https://www.transtats.bts.gov/AverageFare/
https://www.transtats.bts.gov/Data_Elements.aspx?Data=4
https://data.europa.eu/data/datasets/43c6ugqwp92dx7vlgnzja?locale=en
Diversion Airports: https://www.bts.gov/topics/airlines-and-airports/domestic-flights-tarmac-times-more-3-hours-and-international-flights-9 
Taxi out time: https://www.transtats.bts.gov/ONTIME/OriginDestination.aspx
https://www.transtats.bts.gov/ONTIME/Departures.aspx

https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr
https://www.bts.gov/browse-statistical-products-and-data/bts-publications/data-bank-28ds-t-100-domestic-segment-data


How to split into regional, buisness, cargo,...
https://www.eurocontrol.int/sites/default/files/2022-05/eurocontrol-market-segment-update-2022-05.pdf

https://www.easa.europa.eu/eco/eaer/appendix

fuel flow per engine etc.
https://www.easa.europa.eu/en/domains/environment/icao-aircraft-engine-emissions-databank

fuel prices 2022
https://www.kaggle.com/datasets/zusmani/petrolgas-prices-worldwide/data
CC0 license - obtained with google


https://data.transportation.gov/Aviation/Consumer-Airfare-Report-Table-1a-All-U-S-Airport-P/tfrh-tu9e/about_data

https://destinationinsights.withgoogle.com/intl/en_ALL/

Likely not allowed to use, but could be checked if results improve dramatically => warrants manual collection of more data
https://www.kaggle.com/datasets/heitornunes/aircraft-performance-dataset-aircraft-bluebook/data

Idea from Lukas: Add tax dataset?

Idea for a general plan:
1. We go very broad and extensively leverage open datasets
2. We train a small model to find out which features are most important
3. For these features, we do feature engineerin or include additional datasets from the domain
4. We clean the existing data and features to reduce noise
5. We scale up the model to the biggest possible size


---

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

