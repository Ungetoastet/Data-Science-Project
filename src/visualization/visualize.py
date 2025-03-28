import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def plot_delay(df: pd.DataFrame):
    bins = range(0, int(300) + 1, 1)
    plt.hist(df["ArrDelay"], bins=bins)

    # Labels and title
    plt.xlabel("Delay (minutes)")
    plt.ylabel("Number of Flights")
    plt.title("Flight Delays")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()

def plot_distance(df: pd.DataFrame):
    maxmiles = df["Distance"].max()
    bins = range(0, int(maxmiles) + 50, 50)
    plt.hist(df["Distance"], bins=bins)

    # Labels and title
    plt.xlabel("Flight Distance in miles")
    plt.ylabel("Number of Flights")
    plt.title("Flight Distance")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


def plot_top_airports(df: pd.DataFrame, top_n: int):
    unique, count = np.unique(df["DestCityName"], return_counts=True)

    # Create a DataFrame for sorting
    airport_counts = pd.DataFrame({"Airport": unique, "Count": count})
    airport_counts = airport_counts.sort_values(by="Count", ascending=False).head(top_n)

    # Plot
    plt.bar(airport_counts["Airport"], airport_counts["Count"])
    
    plt.xlabel("Destination Airport")
    plt.ylabel("Number of Flights")
    plt.title(f"Top {top_n} Destination Airports")
    plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    
    plt.show()

def plot_top_airlines(df: pd.DataFrame, top_n: int):
    unique, count = np.unique(df["Airline"], return_counts=True)

    # Create a DataFrame for sorting
    airline_counts = pd.DataFrame({"Airline": unique, "Count": count})
    airline_counts = airline_counts.sort_values(by="Count", ascending=False).head(top_n)

    # Plot
    plt.bar(airline_counts["Airline"], airline_counts["Count"])
    
    plt.xlabel("Operating Airline")
    plt.ylabel("Number of Flights")
    plt.title(f"Top {top_n} Airlines")
    plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    
    plt.show()

def plot_yearly_activities(df: pd.DataFrame):
    # Convert to datetime and extract day of year
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["DayOfYear"] = df["FlightDate"].dt.dayofyear

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["DayOfYear"], bins=365)

    # Labels and title
    plt.xlabel("Day of Year")
    plt.ylabel("Number of Flights")
    plt.title("Flight Frequency by Day of Year")
    plt.xticks(range(1, 366, 30))  # Show labels every ~30 days
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()


def plot_winter_delays(df: pd.DataFrame):
    windels = pd.DataFrame({
        "Season": pd.cut(pd.to_datetime(df["FlightDate"]).dt.month, bins=[0, 2, 5, 8, 11, 12], labels=["Winter", "Spring", "Summer", "Autumn", "Winter"], include_lowest=True, ordered=False),
        "Delay": df["ArrDelay"]})
    
    avg_delays = windels.groupby("Season")["Delay"].mean()
    
    plt.figure(figsize=(8, 5))
    avg_delays.plot(kind="bar")
    plt.xlabel("Season")
    plt.ylabel("Average Delay (minutes)")
    plt.title("Average Flight Delay by Season")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_dist_vs_delay(df: pd.DataFrame):
    plt.scatter(df["Distance"], df["ArrDelay"], alpha=0.01)
    plt.ylabel("Delay (minutes)")
    plt.xlabel("Distance (miles)")
    plt.title("Flight Delays: Short-haul vs Long-haul")
    plt.show()


def plot_busy_airport_delays(df: pd.DataFrame, top_n: int):
    airport_counts = df["DestCityName"].value_counts().head(top_n).index
    busy_airport_delays = df[df["DestCityName"].isin(airport_counts)].groupby("DestCityName")["ArrDelay"].mean().sort_values(ascending=False)
    
    busy_airport_delays.plot(kind="bar")
    plt.xlabel("Destination Airport")
    plt.ylabel("Average Delay (minutes)")
    plt.title(f"Average Delay at the {top_n} Busiest Airports")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

if __name__ == "__main__":
    # plot_delay()
    # plot_distance()
    # plot_top_airports(20)
    # plot_top_airlines(20)
    # plot_yearly_activities()
    # plot_winter_delays()
    plot_dist_vs_delay()
    # plot_busy_airport_delays(20)