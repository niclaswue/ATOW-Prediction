import pandas as pd
from utils.dataset import Dataset
from pathlib import Path
from preprocessing.base_preprocessor import BasePreprocessor

# import batchprocessing script
# from preprocessing.trajectory_batchprocessing import create_trajectory_features_batch

root_dir = Path(__file__).parent.parent.absolute()
additional_data_dir = root_dir / "additional_data"
trajectory_data_file = (
    additional_data_dir / "trajectory_features" / "all_trajectory_features.parquet"
)


class TrajectoryPreprocessor(BasePreprocessor):
    def process(self, dataset: Dataset) -> Dataset:
        # check if the additional data directory contains the trajectory data
        if not trajectory_data_file.exists():
            raise ValueError(
                "TrajectoryPreprocessor: Trajectory data not found. Run batch processing script to create the features."
            )

        print("TrajectoryPreprocessor: Loading trajectory data.")
        trajectory_features = pd.read_parquet(trajectory_data_file)

        # check if all flight_ids in the dataset are in the trajectory features
        # if not, run the batch processing script again (this could only happen if the input dataset is changed and new trajectories are available)
        if not set(dataset.df["flight_id"]).issubset(
            set(trajectory_features["flight_id"])
        ):
            print(
                "TrajectoryPreprocessor: Not all flight_ids in the dataset are in the trajectory features. Run batch processing script again to create the features."
            )
            # create_trajectory_features_batch()
            trajectory_features = pd.read_parquet(trajectory_data_file)

        # add the features to the dataset, matching on the flight_id
        dataset.df = dataset.df.merge(trajectory_features, on="flight_id")
        print("TrajectoryPreprocessor: Trajectory features added to dataset.")
        return dataset
