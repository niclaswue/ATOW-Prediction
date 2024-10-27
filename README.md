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
python download_scripts/download_competition_data.py
```

### Additional Data
Next we download the additional datasets that were used to boost the performance. All used datasets and attribution can be found under the Dataset Overview chapter.

```
python download_scripts/download_additional_data.py
```
This will download the simple to fetch tabular datasets to the `additional_data` directory.

We also use daily METAR weather data. Right now, we only download the METARs for the destination airports.
The following script will gather all unique destination airports for each day and download the reports for them.
In the end, they are combined to one large weather dataset. All downloaded links are saved in `processed.txt` which allows you to resume the download if needed.
```
python download_scripts/download_weather_data.py
```

Most airlines use specific configuration in their aircraft, such as different seat layout. This affects the overall weight of the aircraft. 
We therefore tried to add data about airline's fleets, by scraping publicly available aircraft information of different airlines from websites.
```
python download_scripts/scrape_aircraft_info.py
```

For an overview of all the additional datasets see the list of [additional data sources](documentation/additional_data_sources.md).

### Prepare Trajectory Features
For the next step, you will need a separate conda environment with the `traffic` library installed.
```
# Example Installation
conda create -n traffic -c conda-forge python=3.10 traffic
conda activate traffic
```
Otherwise refer to [Traffic docs](https://traffic-viz.github.io/installation.html).

Our model takes some input features from the OSN Trajectories. Running the Preprocessing of the Trajectories can take a while, therefore this is done in a separate step and the result is saved as `all_trajectory_features.parquet` in the `additional_data` directory.
Excpect this to take multiple hours (up to 10 hours on a regular Laptop PC). On a large machine you may be able to use GNU parallel.
```
python ./preprocessing/trajectory_batchprocessing.py
```
> [!TIP]
> If you have a more performant machine, edit the number of parallel processes in the constant `POOL_NUMBER` in `preprocessing/trajectory_batchprocessing.py`

Once all data is downloaded and the trajectory-features are created, put the `all_trajectory_features` under `additional_data/trajectory_features`. Then you can continue with running the training.

### Run the training
Create a free personal account at wandb.ai, then after pip installing wandb log in using `wand login`. Afterwards, you can use the the wandb training. If you do not want to use wandb, execute `wandb disabled`.

To just test out if everything is working fine, run
```python
python run_wandb.py --final
```

for the final submission, run a command similar to this with your chosen time limit (36000 = 10h):
```python
python run_wandb.py --time 36000 --final
```

To submit the results to the leaderboard, you need to place a file .access_keys.json in the root of the repo. The file is the one attached in the initial email sent to participants. Next, enter the model artifact name or local path in `submit.py`. Finally, execute
```
python submit.py
```

### Key Modules and Classes

- **download_competition_data.py**: Handles downloading of the main competition data, including the OSN trajectory files.
- **download_additional_data.py**: Manages downloading of supplementary datasets to enhance model performance.
- **download_weather_data.py**: Gathers METAR weather data for destination airports and compiles it into a comprehensive dataset.
- **scrape_aircraft_info.py**: Scrapes airline alliance data from the for more detailled aircraft information (see [additional data sources](documentation/additional_data_sources)).
- **run_wandb.py**: Main script to initiate the training process, with model information stored in [Weights&Biases](https://wandb.ai) for MLOps.
- **preprocessing directory**: This directory contains the preprocessors used to extract features from the various datasets.

### Additional Documentation

- [**additional_data_sources.md**](documentation/additional_data_sources.md): Lists all additional datasets used, along with their licenses and attributions.
- [**project_overview.md**](documentation/project_overview.md): General project overview

### Data Directories

- **data/**: Contains the primary competition data.
- **additional_data/**: Stores additional datasets that are used to improve model performance.


