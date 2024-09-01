import urllib.request
from pathlib import Path
from tqdm import tqdm

Path("statistics_data").mkdir(exist_ok=True)
Path("runway_data").mkdir(exist_ok=True)


def download(url, out_path, total=1):
    with tqdm(total=total) as pbar:
        hook = lambda a, b, c: pbar.update(1)
        urllib.request.urlretrieve(url, out_path, reporthook=hook)
    print("Done")


URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
out_path = "runway_data/airports.csv"
size = 1442
download(URL, out_path, size)

URL = "https://davidmegginson.github.io/ourairports-data/runways.csv"
out_path = "runway_data/runways.csv"
size = 481
download(URL, out_path, size)

URL = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/avia_tf_apal?format=SDMX-CSV&compressed=false"
out_path = "statistics_data/estat_avia_tf_apal_en.csv"
size = 16159
download(URL, out_path, size)


# TODO
# https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4%20Pn44vr45
# Download month by month T-100 segment data (All carriers) unzip and put all files into statistics data
