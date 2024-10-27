# Project Overview

![](data_flow.png)

## General Overview
Our goal with the challenge was to integrate as many data sources as possible into our dataset and use AutoML for the tedious taks like feature selection, hyperparameter tuning and ensembling. Therefore, the main focus is put on the input data, rather than the mode itself. To be able to effectively iterate on and leverage external datasets, we created a class for each preprocessor. A preprocessor class alters the dataset. Some preprocessors depend on previous preprocessor features, some add external data. For better performance, each preprocessor acts as a smart caching layer using [**joblib**](https://github.com/joblib/joblib). For an identical input, the output is retrieved from cache, once the input changes, the processing is being triggered. This saved us time and headaches during development. For additional performance, we used the functools caching decorator throughout our computations. As for model selection we took inspiration from Kaggle competitions and chose [**Autogluon**](https://github.com/autogluon/autogluon), which is the leading AutoML tool developed by Amazon. Autogluon will try all kinds of gradient boosted trees, sklearn classifiers and neural networks and finally builds a stacked ensemble automatically. Conveniently, we can specify a time limit for the training, the more time we spend, the better the model. Lastly, we wanted to keep track of different versions and store intermediate models, therefore we set up [**Weights & Biases**](https://github.com/wandb/wandb) as an experiment tracking tool. 

<img width="1420" alt="image" src="https://github.com/user-attachments/assets/ae6d1482-a042-4f31-8228-5cc2ef1834f8">

W&B allows us to monitor the training in the browser from anywhere, including plots, tables and logging output. For example, the feature importance of each feature is logged as a table after each training. The full feature importance table of one of the last models can be seen [here](feature_importance.csv). We also log validation set predictions and all metrics like RMSE, MAE etc. Weights & Biases also stores our models as artifacts in the cloud. Here it should be noted that online use of W&B is of course optional for our solution. We intended to write a decoupled logging class to make it modular but we were too limited in time.


## Preprocessor Overview
We now describe all used preprocessors and how the transform the dataset. A comprehensive overview of these features is available [here](dataset_overview.md). 
All preprocessors are described in the order they are used in the final submission.

### AirportPreprocessor
Adds information about the adep or ades airport to the dataset. Specifically, we use data from [ourairports.com](http://ourairports.com/data/) and from [airportsdata](https://github.com/mborsetti/airportsdata/). Some small number of missing airports was researched manually online. Additionally, we add timezone features and distance features.

### OpenAPAircraftPerformancePreprocessor
We use [openap](https://openap.dev/) for aircraft and engine properties. Additional, engines and aircraft not present in openap where researched manually. Values that could not be found where estimated by the Claude language model.

### AircraftPerformancePreprocessor
We found that some important values like MTOW differed very slightly. They of course depend on the specific aircraft in question, however we thought it would be useful to have an alternative data source for aircraft performance. We therefore build a scraper to scrape the pages  "https://oneworldvirtual.org", "https://staralliancevirtual.org" and "https://skyteamvirtual.org". These also include seat configurations for each airline. To determine the airline we manually looked up the routes and aircraft types. This may be controversial, therefore we added a toggle to switch that off.

### FuelPricePreprocessor
We thought that the fuel price might have a small impact on the fuel carried. We therefore used an open dataset from United Nations to add fuel prices. Source: [data.un.org](https://data.un.org/Handlers/DownloadHandler.ashx?DataFilter=cmID:JF&DataMartId=EDATA&Format=csv&c=2,5,6,7,8&s=_crEngNameOrderBy:asc,_enID:asc,yr:desc)

### RunwayInfoPreprocessor
Using ourairports, we added runway information such as runway length and elevation.

### PaxFlowPreprocessor
The passengers on board have the biggest impact on the weight of an aircraft. We therefore add passenger flow features for each airport using [data from the EU](https://ec.europa.eu/eurostat/cache/metadata/en/avia_pa_esms.htm).

### WeatherDataPreprocessor
The weather at the destination airport can determine the amount of fuel that is loaded. We therefore wrote a script (download_weather_data.py) that can fetch the METARs for all destination airports at the time of arrival using the publicly provided API by Iowa State University. This could be extended for departure weather in the future.

### WeatherSafetyFeatures
We use the METAR features to create additional features which should encapsulate how bad the weather at the detination is. These features where calculated using Claude.

### DerivedFeaturePreprocessor
We derived some features from the challenge data, for example day of week or route as a combination of adep and ades.

### TrajectoryPreprocessor
We use the provided OSN trajectory data from the PRC Data Challenge to calculate features about the cruise flight and the initial climb phase, as well as the mean headwind component during cruise.. Additionally, we defined a flag to see which flights contained the takeoff, cruise, and landing part of the trajectory, since not all trajectories were complete. The Preprocessor uses the precalculated features generated by the `trajectory_batchprocessing.py` script (which are saved locally after execution).   

### OpenAPFuelFlowPreprocessor
In addition to the aircraft properties, we also use OpenAP for fuel flow calculation. However, due to limited availability of dragpolar models, we always use an A320 as base. We do however use the measured altitude and TAS in the calculation.

### FeatureEngineeringPreprocessor
We further engineered features to help the model pick up trends by linearizing data. For example the ration of cruise altitude / service ceiling could be more useful than either of them alone.  

### CreativeWeightPreprocessor
Lastly, we asked Claude to come up with creative features that are not commonly thought of.

### CleanDatasetPreprocessor
And finally, we clean the dataset up by removing columns that were consistently ignored by our models. All further cleaning is taken care of by Autogluon.

## Final Submission Dataset
The final submission dataset is once again cleaned, because we found a lot of less important features during our analysis. These feature columns are removed from the dataset with the `CleanDatasetPreprocessor.py`, before the model is trained.   

## Extensibility
We purposefully designed our project to be able to include various data sources and different machine learning models:

1. **New Data Sources**
   - Add new data to the project `additional_data` directory that might be useful for ATOW 
   - Create new preprocessor classes inheriting from `BasePreprocessor`
   - Implement the `process` method to transform raw data into useful features
   - Add the preprocessor to the pipeline configuration in the `run.py` or `run_wandb.py` script

2. **Alternative Models**
   - Implement new model classes inheriting from `BaseModel`
   - Customize training and prediction logic of new model
   - Add model-specific configurations and hyperparameters
  
## What we didnt do
We took great care to not overfit our model on the submission set. We could have specifically filtered our training data to be most similar to the samples in the submission set or we could weight samples differently to boost our score. However, we decided that this would be against the spirit of the challenge as the model has to also work on new and unseen data. 

