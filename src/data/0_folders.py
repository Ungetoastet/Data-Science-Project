import os

def dataset_folders(main_exe:bool = False) -> None:
    """Creates all the dataset folders"""
    
    if main_exe:
        output_path = "../../data/"
    else:
        output_path = "./data/"

    os.makedirs(output_path + "raw", exist_ok=True)
    os.makedirs(output_path + "extracted", exist_ok=True)
    os.makedirs(output_path + "filtered", exist_ok=True)


if __name__ == "__main__":
    dataset_folders(True)
