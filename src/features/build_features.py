import pandas as pd
import numpy as np

def build_airline_rank(df: pd.DataFrame):
    unique, count = np.unique(df["Airline"], return_counts=True)

    airline_counts = pd.DataFrame({"Airline": unique, "Count": count})
    airline_counts = airline_counts.sort_values(by="Count", ascending=False)

    df["AirlineRank"] = df["Airline"].map(
        airline_counts.set_index("Airline")["Count"].rank(method="dense", ascending=False).astype(int)
    )

    return df

def build_dest_airport_rank(df: pd.DataFrame):
    unique, count = np.unique(df["DestCityName"], return_counts=True)

    airport_counts = pd.DataFrame({"Airport": unique, "Count": count})
    airport_counts = airport_counts.sort_values(by="Count", ascending=False)

    df["DestRank"] = df["DestCityName"].map(
        airport_counts.set_index("Airport")["Count"].rank(method="dense", ascending=False).astype(int)
    )
    return df

def build_dep_airport_rank(df: pd.DataFrame):
    unique, count = np.unique(df["OriginCityName"], return_counts=True)

    airport_counts = pd.DataFrame({"Airport": unique, "Count": count})
    airport_counts = airport_counts.sort_values(by="Count", ascending=False)

    df["DepRank"] = df["OriginCityName"].map(
        airport_counts.set_index("Airport")["Count"].rank(method="dense", ascending=False).astype(int)
    )
    return df

def build_day_of_year(df: pd.DataFrame):
    df["DayOfYear"] = pd.to_datetime(df["FlightDate"]).dt.dayofyear
    return df


def build_airport_load(df: pd.DataFrame):
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])

    # Abflüge pro Flughafen & Tag zählen
    dep_counts = df.groupby(["FlightDate", "OriginCityName"]).size().reset_index(name="DepLoad")

    # Ankünfte pro Flughafen & Tag zählen
    arr_counts = df.groupby(["FlightDate", "DestCityName"]).size().reset_index(name="ArrLoad")

    # Die Werte zum Haupt-DataFrame mergen
    df = df.merge(dep_counts, on=["FlightDate", "OriginCityName"], how="left")
    df = df.merge(arr_counts, on=["FlightDate", "DestCityName"], how="left")

    return df
