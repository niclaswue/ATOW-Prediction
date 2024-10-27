from pathlib import Path
import random
from tqdm import tqdm
import pandas as pd

from utils.dataset import Dataset


class DataLoader:
    def __init__(self, data_path: Path, num_days: int = 0, seed: int = 0) -> None:
        self.path = data_path
        # self.num_days = num_days
        random.seed(seed)

    def load(self):
        challenge = self.load_csv("challenge_set.csv")
        submission = self.load_csv("submission_set.csv")
        final_submission = self.load_csv("final_submission_set.csv")
        days = list(self.path.rglob("*.parquet"))
        random.shuffle(days)
        # trajectories = [self.load_parquet(d) for d in tqdm(days[: self.num_days])]
        return challenge, submission, final_submission

    def load_csv(self, csv_file):
        filepath = self.path / Path(csv_file)
        if not filepath.exists():
            return None
        return Dataset(pd.read_csv(filepath), name=csv_file)

    def load_parquet(self, parquet_file):
        df = pd.read_parquet(parquet_file)
        return Dataset(df, name=parquet_file)
