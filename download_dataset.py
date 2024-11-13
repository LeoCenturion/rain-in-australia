import os
import requests
import zipfile

# URL of the dataset
url = 'https://www.kaggle.com/api/v1/datasets/download/jsphyg/weather-dataset-rattle-package'

# Destination file path for the zip file
destination_zip = os.path.expanduser('./data/rains.zip')

# Extract directory for the unzipped contents
extract_dir = os.path.expanduser('./data/')

# Ensure the destination directory for the zip file exists
os.makedirs(os.path.dirname(destination_zip), exist_ok=True)

# Ensure the extraction directory exists
os.makedirs(extract_dir, exist_ok=True)

# Send GET request to download the file
response = requests.get(url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Save the zip file to the destination
    with open(destination_zip, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print(f"Download completed and saved to {destination_zip}")

    # Unzip the file
    try:
        with zipfile.ZipFile(destination_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"File successfully unzipped to {extract_dir}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
else:
    print(f"Failed to download file. HTTP status code: {response.status_code}")
