# src/utils.py
import pandas as pd
from sklearn.datasets import fetch_california_housing

def load_dataset(use_csv=False, csv_path=None):
    if use_csv and csv_path:
        return pd.read_csv(csv_path)
    else:
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame
        df.rename(columns={"MedHouseVal": "PRICE"}, inplace=True)  # Rename target column
        return df