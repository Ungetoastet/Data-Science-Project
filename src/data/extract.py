import zipfile
import os

from tqdm import tqdm

def dataset_extract(main_exe:bool = False) -> None:    
    """Extracts the zip file into a /data/extracted/"""
    if main_exe:
        input_path = "../../data/raw/flight.zip"
        output_path = "../../data/extracted/Combined_Flights_2022.csv"
    else:
        input_path = "../data/raw/flight.zip"
        output_path = "../data/extracted/Combined_Flights_2022.csv"

    if os.path.exists(output_path):
        ans = input("Extracted files found. Skip extraction? (Y/n)")
        if ans.upper() == "Y":
            print("Skipping extraction...")
            return

    with zipfile.ZipFile(input_path, "r") as zr:
        # Leave out unnecessary doubles
        file_list = [f for f in zr.namelist() if f.endswith(".csv") and not f.startswith("raw")]
        total_files = len(file_list)

        with tqdm(total=total_files, unit="file", desc="Extracting", leave=False) as progress_bar:
            for file in file_list:
                zr.extract(file, input_path[:-14] + "extracted")
                progress_bar.update(1)
    print("Done!")

if __name__ == "__main__":
    dataset_extract(True)

