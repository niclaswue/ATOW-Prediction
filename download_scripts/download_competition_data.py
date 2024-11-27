from pathlib import Path
from pyopensky.s3 import S3Client
from tqdm import tqdm

data_dir = Path(__file__).parent.parent / "data"
data_dir.mkdir(exist_ok=True)

s3 = S3Client()
objects = [obj for obj in s3.s3client.list_objects("competition-data", recursive=True)]
objects = [obj for obj in objects if not (data_dir / obj.object_name).exists()]
objects = reversed(objects)  # to get csv files first

for obj in tqdm(objects):
    print(f"Downloading {obj.object_name}")
    s3.download_object(obj, filename=data_dir)
