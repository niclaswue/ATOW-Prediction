import pandas as pd


def add_derived_features(dataset):
    print("Adding derived features")
    # add day of week
    dataset.df["day"] = pd.to_datetime(dataset.df.date).dt.day
    dataset.df["month"] = pd.to_datetime(dataset.df.date).dt.month
    dataset.df["year"] = pd.to_datetime(dataset.df.date).dt.year
    dataset.df["day_of_week"] = pd.to_datetime(dataset.df.date).dt.day_of_week
    print("Done.")
    return dataset
