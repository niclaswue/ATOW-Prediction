from pathlib import Path
import pandas as pd
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import json
import tempfile

from utils.data_loader import DataLoader
from functools import cache

Path("weather_data").mkdir(exist_ok=True)

loader = DataLoader(Path("data"), num_days=0, seed=1337)
challenge, submission, final_submission, _ = loader.load()

concat_dfs = [challenge.df, submission.df]
if final_submission:
    concat_dfs.append(final_submission.df)
airport_df = pd.concat(concat_dfs)
airport_df[["arrival_time", "ades", "name_ades", "country_code_ades"]]

# Constants
MAX_ATTEMPTS = 6
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"


@cache
def download_data(uri):
    print(f"Downloading {uri} ...")
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = requests.get(uri, timeout=300).text
            if data is not None and not data.startswith("ERROR"):
                return data
        except Exception as exp:
            print(f"download_data({uri}) failed with {exp}")
            time.sleep(5)
        attempt += 1
    print("Exhausted attempts to download, returning empty data")
    return ""


def save_airport_weather(row, location=Path("weather_data")):
    processed = location / "processed.txt"

    date = row["arrival_date"]
    airports = sorted(row["ades"])

    start_time = pd.to_datetime(date)
    end_time = start_time + timedelta(hours=23, minutes=59, seconds=59)
    # Construct the API URL
    url = (
        f"{SERVICE}station={','.join(airports)}&year1={start_time.year}"
        f"&month1={start_time.month}&day1={start_time.day}&hour1={start_time.hour}"
        f"&minute1={start_time.minute}&year2={end_time.year}&month2={end_time.month}"
        f"&day2={end_time.day}&hour2={end_time.hour}&minute2={end_time.minute}"
        "&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T&direct=yes&report_type=1,2,3,4,5,6"
    )
    if processed.exists() and url in processed.read_text():
        print(f"URL {url} was already fetched, skipping.")
        return

    data = download_data(url)
    lines = data.strip().split("\n")
    lines = [l for l in lines if not l.startswith("#DEBUG")]

    # to convert it into a pandas dataframe
    tmp = tempfile.NamedTemporaryFile().name
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    df = pd.read_csv(tmp)

    # we save it at provided location
    df.to_parquet(location / f"weather_{date}.parquet")

    # finally we save the URLs already fetched so that we can resume fetching later
    with open(processed, "a") as f:
        f.write(f"{url}\n")


if __name__ == "__main__":
    df = airport_df
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df["arrival_date"] = pd.to_datetime(df["arrival_time"]).dt.date
    date_ades_comb = df.groupby("arrival_date")["ades"].unique().reset_index()
    date_ades_comb.apply(save_airport_weather, axis=1)
