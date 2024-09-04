# ATOW-Prediction

This is our entry for the PRC Data Challenge: https://ansperformance.eu/study/data-challenge

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


### Run the training
To run the training, start:
```
python run.py
```


## Dataset Overview
TODO make a list.
TODO: Provide download links for the preprocessed datasets, otherwise it takes a long time to preprocess the data 

# Structure of the repository
TODO make an overview of the different modules and classes.

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
