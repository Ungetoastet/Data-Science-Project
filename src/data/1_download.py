import requests
import zipfile
import os

from tqdm import tqdm

def dataset_download(main_exe:bool = False) -> None:
    url = "https://www.kaggle.com/api/v1/datasets/download/robikscube/flight-delay-dataset-20182022"
    
    if main_exe:
        output_path = "../../data/raw/flight.zip"
    else:
        output_path = "./data/raw/flight.zip"

    skipdl = False
    if os.path.exists(output_path):
        ans = input("Zip file found. Skip download? (Y/n)")
        if ans.upper() == "Y":
            skipdl = True

    if skipdl:
        print("Skipping download...")
    else:
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

    with zipfile.ZipFile(output_path, "r") as zr:
        # Leave out unnecessary doubles
        file_list = [f for f in zr.namelist() if f.endswith(".csv") and not f.startswith("raw")]
        total_files = len(file_list)

        with tqdm(total=total_files, unit="file", desc="Extracting") as progress_bar:
            for file in file_list:
                zr.extract(file, output_path[:-10])
                progress_bar.update(1)
    print("Done!")

if __name__ == "__main__":
    dataset_download(True)

