import requests
import pandas as pd
from pathlib import Path

path = Path("additional_data") / "aircraft_data"
path.mkdir(parents=True, exist_ok=True)

response = requests.post("https://www4.icao.int/doc8643/External/AircraftTypes")

if response.status_code == 200:
    df = pd.DataFrame(response.json())
    df.to_csv(path / "icao_aircraft_info.tsv", sep="\t", index=False)
    print("Data saved to aircraft_types.tsv")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
