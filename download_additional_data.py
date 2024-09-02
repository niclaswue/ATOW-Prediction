import urllib.request
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import requests
import zipfile


def download(url, out_path, total=1):
    out_path = Path("additional_data") / Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with tqdm(total=total) as pbar:
        hook = lambda a, b, c: pbar.update(1)
        urllib.request.urlretrieve(url, str(out_path), reporthook=hook)
    print("Done")


def download_and_extract_zip(url, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    zip_path = output_dir / "temp.zip"

    # Download and save the zip file
    zip_path.write_bytes(requests.get(url).content)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Delete the zip file
    zip_path.unlink()


URL = "https://data.un.org/Handlers/DownloadHandler.ashx?DataFilter=cmID:JF&DataMartId=EDATA&Format=csv&c=2,5,6,7,8&s=_crEngNameOrderBy:asc,_enID:asc,yr:desc"
out_dir = Path("additional_data") / "airport_data" / "fuel"
download_and_extract_zip(URL, out_dir)
next(out_dir.glob("UNdata_Export*.csv")).rename(out_dir.parent / "UN_fuel_data.csv")
out_dir.rmdir()

URL = "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv"
out_path = "airport_data/country_codes.csv"
size = 4
download(URL, out_path, size)

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
out_path = "runway_data/airports.csv"
size = 1442
download(URL, out_path, size)

URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"
out_path = "runway_data/runways.csv"
size = 481
download(URL, out_path, size)

URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/avia_tf_apal?format=SDMX-CSV&compressed=false"
out_path = "airport_data/estat_avia_tf_apal_en.csv"
size = 16159
download(URL, out_path, size)

# # could be useful to divide pax / flights?
# URL = (
#     "https://www.eurocontrol.int/performance/data/download/csv/airport_traffic_2022.csv"
# )
# out_path = "statistics_data/airport_traffic_2022.csv"
# size = 882
# download(URL, out_path, size)

# # could indicate additional taxi fuel when usually long taxi
# URL = "https://www.eurocontrol.int/performance/data/download/csv/taxi_out_additional_time_2022.csv"
# out_path = "statistics_data/taxi_out_additional_time_2022.csv"
# size = 12
# download(URL, out_path, size)

# # additional mins before can land, could indicate if more fuel was taken out
# URL = "https://www.eurocontrol.int/performance/data/download/csv/asma_additional_time_2022.csv"
# out_path = "statistics_data/asma_additional_time_2022.csv"
# size = 12
# download(URL, out_path, size)

# # could maybe use more from here: https://ansperformance.eu/csv/

# TODO
# https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45
# Download month by month T-100 segment data (All carriers) unzip and put all files into statistics data