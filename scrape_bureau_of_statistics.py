from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
from tqdm import tqdm
import time
import zipfile

# time between requests (shorter may be possible, but we want to be nice)
TIMEOUT = 60  # seconds


def setup_driver(download_directory):
    chrome_options = Options()
    prefs = {
        "download.default_directory": str(download_directory.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    print("Installing Chrome...")
    service = Service(ChromeDriverManager().install())
    print("Done.")
    return webdriver.Chrome(service=service, options=chrome_options)


def wait_for_element(driver, by, value, timeout=60):
    print()
    print(f"Waiting for {value}")
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )


def wait_for_download_to_start(download_directory, timeout=120):
    end_time = time.time() + timeout
    while time.time() < end_time:
        files = list(download_directory.glob("*.zip"))
        if files:
            return files[0]  # Return the first downloaded file
        time.sleep(1)
    raise TimeoutException("Download did not start within the given time frame.")


def rename_and_extract(downloaded_file, download_directory, year, month):
    new_filename = f"Data_{year}_{month}.zip"
    new_filepath = download_directory / new_filename
    downloaded_file.rename(new_filepath)

    # Extract the ZIP file
    extract_dir = download_directory / f"{year}_{month}"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(new_filepath, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    new_filepath.unlink()
    print(f"Extracted {new_filename} to {extract_dir}")


def scrape_data(url, download_directory):
    driver = setup_driver(download_directory)
    print("Fetching page...")
    driver.get(url)
    print("Done.")

    try:
        check_all = wait_for_element(driver, By.ID, "chkAllVars")
        if not check_all.is_selected():
            check_all.click()
            time.sleep(TIMEOUT)

        month_dropdown = wait_for_element(driver, By.ID, "cboPeriod")
        year_dropdown = wait_for_element(driver, By.ID, "cboYear")

        month_options = [
            option.get_attribute("value")
            for option in month_dropdown.find_elements(By.TAG_NAME, "option")
            if option.get_attribute("value") != "All"
        ]
        year_options = [
            option.get_attribute("value")
            for option in year_dropdown.find_elements(By.TAG_NAME, "option")
        ]
        existing_dirs = list(download_directory.glob("*_*"))
        existing = [d.name.split("_") for d in existing_dirs if d.is_dir()]
        print("Starting...")
        for year in tqdm(year_options):
            for month in tqdm(month_options):
                if [year, month] in existing:
                    print(f"Already downloaded {year}-{month}. Skipping")
                    continue
                try:
                    time.sleep(TIMEOUT)
                    Select(year_dropdown).select_by_value(year)
                    Select(month_dropdown).select_by_value(month)

                    download_button = wait_for_element(driver, By.ID, "btnDownload")
                    download_button.click()

                    downloaded_file = wait_for_download_to_start(download_directory)
                    rename_and_extract(downloaded_file, download_directory, year, month)
                except Exception as e:
                    print("Received error:")
                    print(e.__repr__())
                    print("Skipping.")
                    time.sleep(TIMEOUT * 10)

    except Exception as e:
        print(f"A fatal error occurred: {e}")

    finally:
        driver.quit()


if __name__ == "__main__":
    url = "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMG&QO_fu146_anzr=Nv4+Pn44vr45"
    download_directory = Path("additional_data/T100_data")
    download_directory.mkdir(exist_ok=True)

    scrape_data(url, download_directory)
