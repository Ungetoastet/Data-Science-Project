import requests
import os

from tqdm import tqdm

def dataset_download(main_exe:bool = False, auto_skip = False) -> None:
    """Downloads the dataset to /data/raw"""
    url = "https://www.kaggle.com/api/v1/datasets/download/robikscube/flight-delay-dataset-20182022"
    
    if main_exe:
        output_path = "../../data/raw/flight.zip"
    else:
        output_path = "../data/raw/flight.zip"

    if os.path.exists(output_path):
        if auto_skip:
            print("Auto Skipping Download...")
            return
        else:
            ans = input("Zip file found. Skip download? (Y/n)")
            if ans.upper() == "Y":
                print("Skipping download...")
                return

    chunk_size = 1024  # 1KB

    response = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    print(f"Downloaded to {output_path}")

if __name__ == "__main__":
    dataset_download(True)

