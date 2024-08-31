from pathlib import Path
import json
from pyopensky.s3 import S3Client
from pyopensky.config import opensky_config_dir
from tqdm import tqdm

print(opensky_config_dir)

# SECRETS_FILE = ".access_keys.json"

# if not Path(SECRETS_FILE).exists():
#     print(f"The file {SECRETS_FILE} does not exist.")
#     print(f"Please paste the provided access keys into a file called {SECRETS_FILE}")
#     exit()

# access_keys = json.load(open(SECRETS_FILE, "r"))
# key = access_keys.get("bucket_access_key")
# secret = access_keys.get("bucket_access_secret")

Path("data").mkdir(exist_ok=True)

s3 = S3Client()
objects = [obj for obj in s3.s3client.list_objects("competition-data", recursive=True)]
objects = [obj for obj in objects if not (Path("data") / obj.object_name).exists()]
objects = reversed(objects)  # to get csv files first

for obj in tqdm(objects):
    print(f"Downloading {obj.object_name}")
    s3.download_object(obj, filename=Path("data"))
