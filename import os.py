import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize and authenticate
api = KaggleApi()
api.authenticate()

# Download the dataset as a zip file (no auto-extraction)
dataset = "afterjacob/5k-podcast-conversations-tags-and-embeddings"
download_dir = "downloads"
os.makedirs(download_dir, exist_ok=True)

api.dataset_download_files(dataset, path=download_dir, unzip=False)

print(f"Downloaded zip file to: {os.path.join(download_dir, '5k-podcast-conversations-tags-and-embeddings.zip')}")
