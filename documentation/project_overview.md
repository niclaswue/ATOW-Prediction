# Project Overview
This project implements a flexible and extensible system for estimating Aircraft Take-Off Weight (ATOW) using a modular architecture that combines various data sources and machine learning approaches into a customizable data pipeline. The system is designed with adaptability and maintainability in mind, allowing for easy integration of new data sources and modeling techniques.


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

## Implementation Features

1. **Modular Preprocessing Pipeline**
   - Built on a `BasePreprocessor` abstract class that enables consistent feature engineering
   - Allows for integration of multiple data sources and feature generation strategies
   - Preprocessors can be easily added, removed, or modified through configuration

2. **Flexible Model Interface**
   - Implements a `BaseModel` abstract class that standardizes model interactions
   - Supports various AI/ML strategies through a common interface
   - Core functionality includes:
     - Training interface (`train`)
     - Prediction capability (`predict`)
     - Model information retrieval (`info`)

3. **Configurable Training Pipeline**
   - Adjustable training/validation split ratios
   - Support for different quality presets and time limits
   
4. **Evaluation Framework**
   - Dedicated metrics evaluation system
   - Support for model performance tracking
   - Feature importance analysis capabilities
   - Integration with Weights & Biases (wandb) for experiment tracking


## Extensibility

We purposefully designed our project to be able to include various data sources and different machine learning models. This makes it easy to include new data or test different AI models in the future and experiment with new features that affect ATOW. We hope that 
The system is designed to be extended in several ways:

1. **New Data Sources**
   - Add new data to the project `additional_data` directory that might be useful for ATOW 
   - Create new preprocessor classes inheriting from `BasePreprocessor`
   - Implement the `process` method to transform raw data into useful features
   - Add the preprocessor to the pipeline configuration in the `run.py` or `ron_wandb.py` script

2. **Alternative Models**
   - Implement new model classes inheriting from `BaseModel`
   - Customize training and prediction logic while maintaining a consistent and easy interface
   - Add model-specific configurations and hyperparameters
  

