import requests
from bs4 import BeautifulSoup
import json
import time
import random
from pathlib import Path
from tqdm import tqdm


def get_soup(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    return BeautifulSoup(response.content, "html.parser")


def get_fleet_type_links(base_url):
    soup = get_soup(base_url + "/fleet/types")
    links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("/fleet/types/")
    ]
    return list(set(links))


def get_fleet_model_links(base_url, type_link):
    soup = get_soup(base_url + type_link)
    links = [
        a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].startswith("/fleet/models/")
    ]
    return list(set(links))


def parse_specifications(soup):
    specs = {}
    spec_table = soup.find("h5", string="Specifications").find_next("table")
    for row in spec_table.find_all("tr"):
        key = row.find("td").text.strip()
        value = row.find_all("td")[1].text.strip()
        specs[key] = value
    return specs


def parse_model_page(full_url, type_name):
    soup = get_soup(full_url)

    name = soup.find("h1").text.strip()
    specs = parse_specifications(soup)
    return {
        "name": name,
        "type": type_name,
        "specifications": specs,
        "source": full_url,
    }


def scrape_fleet_data(base_url):
    fleet_data = []
    type_links = get_fleet_type_links(base_url)

    for type_link in tqdm(type_links):
        type_name = get_soup(base_url + type_link).find("h1").text.strip()
        model_links = get_fleet_model_links(base_url, type_link)

        for model_link in model_links:
            # tqdm.set_description(desc=f"Processing model: {model_link}")
            full_url = base_url + model_link
            fleet_data.append(parse_model_page(full_url, type_name))

            # Be nice to the server
            time.sleep(random.uniform(1, 3))

    return fleet_data


def main():
    dataset = []
    for url in [
        "https://oneworldvirtual.org",
        "https://staralliancevirtual.org",
        "https://skyteamvirtual.org",
    ]:
        dataset += scrape_fleet_data(url)

    for type_name, url in [
        (
            "A310",
            "https://staralliancevirtual.org/fleet/models/turkish-airlines-airbus-a310-308-f",
        ),
        (
            "A310",
            "https://oneworldvirtual.org/fleet/models/royal-jordanian-airbus-a310-304-f",
        ),
        (
            "B737",
            "https://staralliancevirtual.org/fleet/models/turkish-airlines-boeing-737-752-wl",
        ),
        (
            "B752",
            "https://oneworldvirtual.org/fleet/models/aer-lingus-boeing-757-2q8-wl",
        ),
    ]:
        dataset += [parse_model_page(url, type_name)]

    save_dir = Path(__file__).parent / "additional_data" / "aircraft_data"
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "aircraft_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print("Data saved to aircraft_info.json")


if __name__ == "__main__":
    main()
