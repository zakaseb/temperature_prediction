import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response
import matplotlib.pyplot as plt
import io
import base64

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)


# Load the dataset (for creative endpoints) and precompute date features
def load_data(data_path):
    df = pd.read_csv(data_path, parse_dates=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    return df


df = load_data(config["DATA_PATH"])


# Load the trained model
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model(config["MODEL_PATH"])

# Create the FastAPI app
app = FastAPI(title=config["API_TITLE"], version=config["API_VERSION"])


@app.get("/predict/")
def predict_temperature(date: str):
    """
    Predict the mean temperature for a given date in YYYYMMDD format.
    """
    try:
        date_obj = datetime.strptime(date, "%Y%m%d")
        features = np.array([[date_obj.year, date_obj.month, date_obj.day]])
        prediction = model.predict(features)[0]
        return {"date": date, "predicted_mean_temp": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")


@app.get("/monthly_avg/")
def monthly_avg_temperature(year: int, month: int):
    """
    Return the average mean temperature for a specified month and year.
    """
    monthly_data = df[(df['year'] == year) & (df['month'] == month)]
    if monthly_data.empty:
        raise HTTPException(status_code=404, detail="Data not available for the specified month and year.")
    avg_temp = monthly_data['mean_temp'].mean()
    return {"year": year, "month": month, "average_mean_temp": avg_temp}


@app.get("/temperature_trend/")
def temperature_trend(start_date: str, end_date: str):
    """
    Return a temperature trend plot (as a base64 encoded PNG image)
    between start_date and end_date (in YYYYMMDD format).
    This creative endpoint provides a visual summary of temperature trends.
    """
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Date format error: {e}")

    # Filter data for the given date range
    trend_data = df[(df['date'] >= start) & (df['date'] <= end)]
    if trend_data.empty:
        raise HTTPException(status_code=404, detail="No data available for the given date range.")

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(trend_data['date'], trend_data['mean_temp'], marker='o', linestyle='-')
    plt.title("Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.tight_layout()

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    # Encode image to base64 string
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"start_date": start_date, "end_date": end_date, "temperature_trend_plot": img_base64}

@app.get("/")
def read_root():
    return {"message": "Welcome to the London Temperature Prediction API! Visit /docs for the API documentation."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config["HOST"], port=config["PORT"])
