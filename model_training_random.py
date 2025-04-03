import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)


def load_data(data_path):
    """
    Load and preprocess the dataset.
    Assumes the CSV has a 'date' column and a 'mean_temp' column.
    Drops rows where 'mean_temp' is missing.
    """
    df = pd.read_csv(data_path, parse_dates=['date'])
    df['mean_temp'].fillna(df['mean_temp'].mean(), inplace=True)
    # Extract useful date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df


def train_model(df):
    """
    Train a Random Forest model to predict the mean temperature.
    Uses year, month, and day as features.
    """
    X = df[['year', 'month', 'day']]
    y = df['mean_temp']

    # Split data into training and test sets for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model to disk
    with open(config["MODEL_PATH"], "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved at:", config["MODEL_PATH"])


if __name__ == "__main__":
    # Load data and train the model
    df = load_data(config["DATA_PATH"])
    train_model(df)
