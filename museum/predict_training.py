import wandb
from pathlib import Path
from autogluon.tabular import TabularDataset, TabularPredictor
from utils.data_loader import DataLoader
from run_wandb import PREPROCESSORS

SUBMIT_ARTIFACT = "model:v11"

wandb.init(project="flying_penguins", name=f"predict_{SUBMIT_ARTIFACT}")
artifact = wandb.run.use_artifact(SUBMIT_ARTIFACT)
model = TabularPredictor.load(artifact.download())

loader = DataLoader(Path("data"), num_days=0)
dataset, _, _ = loader.load()

for preprocessor in PREPROCESSORS:
    dataset = preprocessor.apply(dataset)

print("Starting prediction")
predictions = model.predict(TabularDataset(dataset.df))
result = dataset.df[["flight_id", "tow"]]
result["tow"] = predictions
result = result.set_index("flight_id")
result.to_csv("training_predictions.csv")
