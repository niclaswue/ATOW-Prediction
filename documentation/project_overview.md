# Project Overview
Our project combines various data sources into a customizable data pipeline for estimating ATOW. The system is designed with adaptability and maintainability in mind, allowing for easy integration of new data sources and modeling techniques.


## Data Pipeline Overview
TODO für Niclas -> Deine Grafik?



## Data Sources and Feature Engineering
Our current pipeline for estimating ATOW incorporates multiple specialized preprocessors for different aspects of flight operations:

- General Airport Information
- Runway Characteristics
- Weather Conditions (METAR)
- Weather Safety Features
- Trajectory Data
- OpenAP Fuel Flow Calculations
- Aircraft Data  
- Passenger Flow Metrics
- Custom Feature Engineering

The final dataset has a number of features in addition to the features provided by the PRC Data Challenge training/challenge data. A comprehensive overview of these features is available [here](dataset_overview.md).

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
  

