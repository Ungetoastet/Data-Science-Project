import os
import pandas as pd
from tqdm import tqdm

def dataset_filter(chunk_size: int = 100_000, main_exe=False, auto_skip=False) -> None:
    """Filters out unnecessary coloumns to reduce file size"""

    columns = ["FlightDate", "Airline", "Cancelled", "Diverted", "CRSDepTime", "DepTime", "DepDelay", "ArrTime", "ArrDelay", "AirTime", "CRSElapsedTime", "ActualElapsedTime", "Distance", "DayOfWeek", "Operating_Airline", "Tail_Number", "Flight_Number_Operating_Airline", "OriginCityName", "OriginStateName", "DestCityName", "DestStateName", "CRSArrTime"]

    if main_exe:
        input_folder = "../../data/extracted/"
        output_folder = "../../data/filtered/"
    else:
        input_folder = "../data/extracted/"
        output_folder = "../data/filtered/"

    if os.path.exists(output_folder + "Combined_Flights_2022.csv"):
        if auto_skip:
            print("Auto Skipping filtration...")
            return
        else:
            ans = input("Filtered files found. Skip filtration? (Y/n)")
            if ans.upper() == "Y":
                print("Skipping filtration...")
                return

    files = [
        "Combined_Flights_2018.csv",
        "Combined_Flights_2019.csv",
        "Combined_Flights_2020.csv",
        "Combined_Flights_2021.csv",
        "Combined_Flights_2022.csv",
    ]

    for i in range(len(files)):
        file = files[i]
        input_file = input_folder + file
        output_file = output_folder + file

        chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)

        total_rows = sum(1 for _ in open(input_file)) - 1  # subtract 1 for the header
        total_chunks = total_rows // chunk_size + 1
        
        with open(output_file, "w", newline="") as f_out:
            for i, chunk in tqdm(enumerate(chunk_iter), total=total_chunks, desc=f"Filtering file {i+1}/{len(files)}", leave=False):
                selected_columns = chunk[columns]
                if i == 0:
                    selected_columns.to_csv(f_out, index=False, header=True)  # Write header for the first chunk
                else:
                    selected_columns.to_csv(f_out, index=False, header=False)  # Skip header for subsequent chunks


if __name__ == "__main__":
    dataset_filter(main_exe=True)
