import pandas as pd
from utils.dataset import Dataset
from tqdm import tqdm
from functools import cache

print("Loading statistics..")
# https://ec.europa.eu/eurostat/cache/metadata/en/avia_pa_esms.htm
sdf = pd.read_csv("statistics_data/estat_avia_tf_apal_en.csv")
sdf.drop(columns=["OBS_FLAG", "DATAFLOW", "LAST UPDATE", "freq"], inplace=True)

sdf["TIME_PERIOD"] = sdf["TIME_PERIOD"].astype(str)
sdf["rep_airp"] = sdf["rep_airp"].astype(str)

# to make it faster, if another year add condition
sdf = sdf[sdf["TIME_PERIOD"].str.startswith("2022-", na=False)]


@cache
def _data_for_ap(airport):
    result = sdf[sdf.rep_airp == airport]
    if len(result) == 0:
        print(f"NO data for AP: {airport}")
    return result


@cache
def get_statistic_data(year, month, airport):
    stats = _data_for_ap(airport)
    stats = stats[stats["TIME_PERIOD"] == f"{year}-{month:0>2}"]
    if len(stats) == 0:
        return stats

    df_pivot = stats.pivot_table(
        index=None,
        columns=["unit", "tra_meas", "airline"],
        values="OBS_VALUE",
    )

    df_pivot.columns = [
        "{}_{}_{}".format(col[0], col[1], col[2]) for col in df_pivot.columns
    ]

    df_pivot.reset_index(drop=True, inplace=True)
    return df_pivot


def add_statistics_data(dataset: Dataset):

    # add day of week
    dataset.df["day_of_week"] = pd.to_datetime(dataset.df.date).dt.day_of_week

    dataset.df["month"] = pd.to_datetime(dataset.df.date).dt.month
    dataset.df["year"] = pd.to_datetime(dataset.df.date).dt.year

    dataset.df["full_adep"] = dataset.df["country_code_adep"] + "_" + dataset.df["adep"]
    dataset.df["full_ades"] = dataset.df["country_code_ades"] + "_" + dataset.df["ades"]

    months = [
        (int(y), int(m))
        for y in dataset.df.year.unique()
        for m in dataset.df.month.unique()
    ]
    for mode in ["adep", "ades"]:
        for ap in tqdm(dataset.df[f"full_{mode}"].unique()):
            if not ap in sdf.rep_airp.unique():
                continue
            for y, m in months:
                stats = get_statistic_data(y, m, ap)
                if len(stats) == 0:
                    print(f"No data for {ap}")
                    continue
                stats = stats.add_prefix(f"stats_{mode}_")
                stats = stats.iloc[0].to_dict()
                for col, val in stats.items():
                    dataset.df.loc[
                        (dataset.df[f"full_{mode}"] == ap)
                        & (dataset.df["year"] == y)
                        & (dataset.df["month"] == m),
                        col,
                    ] = val

    return dataset
