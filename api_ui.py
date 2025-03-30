import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import io
from PIL import Image
import gradio as gr

# Load configuration from config.json
with open("config.json", "r") as f:
    config = json.load(f)


# Load the dataset and add date features
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


# Function for temperature prediction
def predict_temperature_ui(date: str):
    try:
        date_obj = datetime.strptime(date, "%Y%m%d")
    except Exception as e:
        return f"Error: Date format must be YYYYMMDD. {e}"
    features = np.array([[date_obj.year, date_obj.month, date_obj.day]])
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        return f"Prediction error: {e}"
    return f"Predicted mean temperature for {date}: {prediction:.2f}"


# Function for monthly average temperature
def monthly_avg_temperature_ui(year: int, month: int):
    monthly_data = df[(df['year'] == year) & (df['month'] == month)]
    if monthly_data.empty:
        return f"No data available for {year}-{month:02d}."
    avg_temp = monthly_data['mean_temp'].mean()
    return f"Average mean temperature for {year}-{month:02d}: {avg_temp:.2f}"


# Function for temperature trend plot
def temperature_trend_ui(start_date: str, end_date: str):
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except Exception as e:
        return None, f"Date format error: {e}"
    trend_data = df[(df['date'] >= start) & (df['date'] <= end)]
    if trend_data.empty:
        return None, "No data available for the given date range."

    plt.figure(figsize=(10, 5))
    plt.plot(trend_data['date'], trend_data['mean_temp'], marker='o', linestyle='-')
    plt.title("Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return image, "Temperature trend generated."


# Dispatcher function to handle requests
def handle_request(endpoint, date, year, month, start_date, end_date):
    default_image = Image.new("RGB", (1, 1), (255, 255, 255))  # White placeholder image

    if endpoint == "Predict Temperature":
        text = predict_temperature_ui(date)
        return text, default_image
    elif endpoint == "Monthly Average Temperature":
        text = monthly_avg_temperature_ui(year, month)
        return text, default_image
    elif endpoint == "Temperature Trend":
        image, msg = temperature_trend_ui(start_date, end_date)
        if image is None:
            return msg, default_image
        else:
            return msg, image
    else:
        return "Unknown endpoint.", default_image


# Build the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## London Temperature Prediction API")
    gr.Markdown("Select an operation and fill in the required fields below.")

    with gr.Row():
        endpoint_radio = gr.Radio(
            label="Select Endpoint",
            choices=[
                "Predict Temperature",
                "Monthly Average Temperature",
                "Temperature Trend"
            ],
            value="Predict Temperature"
        )

    with gr.Row():
        date_input = gr.Textbox(label="Date (YYYYMMDD)", placeholder="e.g. 20200115")
    with gr.Row():
        year_input = gr.Number(label="Year", value=2020)
        month_input = gr.Number(label="Month", value=1, precision=0)
    with gr.Row():
        start_date_input = gr.Textbox(label="Start Date (YYYYMMDD)", placeholder="e.g. 20200101")
        end_date_input = gr.Textbox(label="End Date (YYYYMMDD)", placeholder="e.g. 20200131")

    text_output = gr.Textbox(label="Result")
    image_output = gr.Image(label="Temperature Trend", type="pil")

    run_button = gr.Button("Submit")
    run_button.click(
        fn=handle_request,
        inputs=[endpoint_radio, date_input, year_input, month_input, start_date_input, end_date_input],
        outputs=[text_output, image_output]
    )

    gr.Markdown("### Instructions:")
    gr.Markdown("- **Predict Temperature:** Enter a date in YYYYMMDD format.")
    gr.Markdown("- **Monthly Average Temperature:** Enter the year and month.")
    gr.Markdown("- **Temperature Trend:** Enter the start and end dates in YYYYMMDD format.")

# Launch the Gradio app
demo.launch(share=True)