import boto3
from pathlib import Path
import json

submission_dir = Path("submissions")
version = 8

SECRETS_FILE = ".access_keys.json"
if not Path(SECRETS_FILE).exists():
    print(f"The file {SECRETS_FILE} does not exist.")
    print(f"Please paste the provided access keys into a file called {SECRETS_FILE}")
    exit()

access_keys = json.load(open(SECRETS_FILE, "r"))

team = access_keys.get("team_name")
team_id = access_keys.get("team_id")
filename = f"{team}_v{version}_{team_id}.csv"


# Create a session using credentials
session = boto3.Session(
    aws_access_key_id=access_keys.get("bucket_access_key"),
    aws_secret_access_key=access_keys.get("bucket_access_secret"),
)

print("Uploading...")
# Create an S3 client with the appropriate configuration
s3_client = session.client("s3", endpoint_url="https://s3.opensky-network.org")

with open(submission_dir / filename, "rb") as data:
    s3_client.upload_fileobj(data, "submissions", filename)
print("Submitted the file.")
