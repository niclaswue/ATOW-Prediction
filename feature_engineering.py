# import sys
# sys.path.append('../')
import warnings
import os
import numpy as np

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.datasets import fetch_california_housing
from openfe import OpenFE, tree_to_formula, transform
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from autogluon.tabular import TabularDataset, TabularPredictor


def convert_dtypes(df):
    # Convert date and time fields to integer timestamps
    date_columns = ["date", "actual_offblock_time", "arrival_time"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df[col] = (
            df[col].astype(np.int64) // 10**9
        )  # Convert to Unix timestamp (seconds since epoch)

    # Convert string columns to category, then to integer codes
    category_columns = [
        "callsign",
        "adep",
        "name_adep",
        "country_code_adep",
        "ades",
        "name_ades",
        "country_code_ades",
        "aircraft_type",
        "wtc",
        "airline",
    ]
    for col in category_columns:
        df[col] = df[col].astype("category")
        df[col] = df[col].cat.codes  # Convert categories to integer codes

    return df


def get_score(train_x, test_x, train_y, test_y):
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=1
    )
    params = {"n_estimators": 10000, "n_jobs": n_jobs, "seed": 1, "verbosity": -1}
    gbm = lgb.LGBMRegressor(**params)
    gbm.fit(
        train_x,
        train_y,
        eval_set=[(val_x, val_y)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = pd.DataFrame(gbm.predict(test_x), index=test_x.index)
    score = mean_squared_error(test_y, pred)
    return score


if __name__ == "__main__":
    n_jobs = os.cpu_count()
    data = pd.read_parquet("ofe_input.parquet")
    data = convert_dtypes(data)

    label = data[["tow"]]
    del data["tow"]

    train_x, test_x, train_y, test_y = train_test_split(
        data, label, test_size=0.2, random_state=1
    )
    # get baseline score
    print("Training...")
    score = get_score(train_x, test_x, train_y, test_y)
    print("The MSE before feature generation is", score)
    # feature generation
    ofe = OpenFE()
    ofe.fit(data=train_x, label=train_y, n_jobs=n_jobs)

    # OpenFE recommends a list of new features. We include the top 10
    # generated features to see how they influence the model performance
    train_x, test_x = transform(
        train_x, test_x, ofe.new_features_list[:10], n_jobs=n_jobs
    )
    score = get_score(train_x, test_x, train_y, test_y)
    print("The MSE after feature generation is", score)
    print("The top 10 generated features are")
    for feature in ofe.new_features_list[:10]:
        print(tree_to_formula(feature))
