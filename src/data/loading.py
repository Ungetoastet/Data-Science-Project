import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

DATA_DIR = Path("../data/filtered")
FLIGHT_FILES = [
    "Combined_Flights_2018.csv",
    "Combined_Flights_2019.csv",
    "Combined_Flights_2020.csv",
    "Combined_Flights_2021.csv",
    "Combined_Flights_2022.csv",
]

def load_single(y:int, chunk_size: int = 100000) -> tuple[pd.DataFrame, dict]:
    """Loads only the flights data from the given year in chunks, with a progress bar."""
    print(f"Loading subset for {y}...")
    
    # Create iterator to load data in chunks
    chunk_iter = pd.read_csv(DATA_DIR / f"Combined_Flights_{y}.csv", chunksize=chunk_size)
    total_rows = sum(1 for _ in open(DATA_DIR / f"Combined_Flights_{y}.csv"))
    df_chunks = []
    
    with tqdm(total=total_rows, unit="rows", desc=f"Loading {y}", leave=False) as progress_bar:
        for chunk in chunk_iter:
            df_chunks.append(chunk)
            progress_bar.update(len(chunk))
    
    df = pd.concat(df_chunks, ignore_index=True)
    print(f"Loaded {len(df)} flights from {y}!")
    
    return df

def load_all(chunk_size: int = 100000) -> tuple[pd.DataFrame, dict]:
    """Loads all flights data into a single DataFrame in chunks, with a progress bar."""
    print("Loading full dataset...")
    
    # List to store chunks
    dfs = []
    
    # Create a progress bar for total rows across all files
    total_rows = sum(sum(1 for _ in open(DATA_DIR / file)) for file in FLIGHT_FILES)
    
    with tqdm(total=total_rows, unit="rows", desc="Loading all years", leave=False) as progress_bar:
        for file in FLIGHT_FILES:
            chunk_iter = pd.read_csv(DATA_DIR / file, chunksize=chunk_size)
            for chunk in chunk_iter:
                dfs.append(chunk)
                progress_bar.update(len(chunk))
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} flights!")
    
    return df

if __name__ == "__main__":
    df = load_all()
    print(f"Coloumns in file: {df.columns.tolist()}")
    print(f"Random entry:")
    [print(f"{col}: {val}") for col, val in df.iloc[random.randint(0, len(df)-1)].items()]
