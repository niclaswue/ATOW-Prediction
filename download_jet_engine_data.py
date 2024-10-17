import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

additional_data_dir = Path("additional_data")


def download_parse_html_table(url):
    # Download the HTML content
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table
    table = soup.find("table")

    # Extract table headers
    headers = [
        "Manufacturer",
        "Model",
        "Application(s)",
        "Thrust (dry) [lbf]",
        "Thrust (wet) [lbf]",
        "SFC (dry) [lb/lbf hr]",
        "SFC (wet) [lb/lbf hr]",
        "Airflow (static) [lb/s]",
        "OPR (static)",
        "FPR (static)",
        "BPR (static)",
        "Thrust (cruise) [lbf]",
        "SFC (cruise) [lb/lbf hr]",
        "Cruise Speed [Mach]",
        "Cruise Altitude [ft]",
        "TIT [K]",
        "Number Spools",
        "Fan Stages",
        "LPC Stages",
        "HPC Stages",
        "HPT Stages",
        "IPT Stages",
        "LPT Stages",
        "Fan Diameter [in]",
        "Length [in]",
        "Width/Diameter [in]",
        "Dry Weight [lb]",
    ]

    # adapt to the page:

    # Extract table data
    data = []
    for row in table.find_all("tr")[1:]:  # Skip the header row
        row_data = [td.text.strip() for td in row.find_all("td")]
        if row_data:
            data.append(row_data[1:])

    data = data[6:-22]  # ignore heading and footer rows etc
    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Clean up the DataFrame
    df = df.replace("", pd.NA)
    df = df.replace("-", pd.NA)
    df = df.replace("", pd.NA).dropna(how="all", axis=1)  # Remove empty columns
    df = df.dropna(how="all", axis=0)  # Remove empty rows

    return df


# Usage example
if __name__ == "__main__":
    print("Downloading engine data...")
    url = "https://www.jet-engine.net/civtfspec_files/sheet001.htm"
    df = download_parse_html_table(url)

    path = Path("additional_data") / "aircraft_data"
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "engines.tsv", sep="\t")
    print("Done.")
