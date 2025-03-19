import zipfile

from tqdm import tqdm

def dataset_extract(main_exe:bool = False) -> None:    
    """Extracts the zip file into a /data/extracted/"""
    if main_exe:
        output_path = "../../data/raw/flight.zip"
    else:
        output_path = "./data/raw/flight.zip"

    with zipfile.ZipFile(output_path, "r") as zr:
        # Leave out unnecessary doubles
        file_list = [f for f in zr.namelist() if f.endswith(".csv") and not f.startswith("raw")]
        total_files = len(file_list)

        with tqdm(total=total_files, unit="file", desc="Extracting") as progress_bar:
            for file in file_list:
                zr.extract(file, output_path[:-14] + "extracted")
                progress_bar.update(1)
    print("Done!")

if __name__ == "__main__":
    dataset_extract(True)

