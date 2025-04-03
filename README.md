# London Temperature Prediction API

## Overview
This project predicts the mean temperature in London using historical weather data. It includes:
- Two model training scripts:
  - `model_training_regression.py` for training a linear regression model.
  - `model_training_random.py` for training a random forest model.
- Two API scripts:
  - `api.py` for the API without a GUI.
  - `api_ui.py` for the API with a browser-based GUI.
- A configuration file (`config.json`) with all necessary settings.
- A `requirements.txt` file with the required libraries.
- The dataset `london_weather.csv` used for training.

## Project Structure

\
├── config.json                \
├── requirements.txt          \
├── london_weather.csv (not included in repo) \
├── model_training_regression.py  (linear regression model)\
├── model_training_random.py   (random forest model)   \
├── temperature_model.pkl (not included in repo)    \
├── api.py (Headless API)                   \
└── api_ui.py    (Gradio GUI enabled API)             

## Prerequisites
- Python 3.8+


## Installation
1. Clone the repository: ` git clone https://github.com/zakaseb/temperature_prediction.git`
2. Navigate to the project directory: `cd <project-directory>`
3. Create and activate a virtual environment:
- On Linux/Mac:
  ```
  python -m venv venv
  source venv/bin/activate
  ```
- On Windows:
  ```
  python -m venv venv
  venv\Scripts\activate
  ```
4. Install the required libraries: `pip install -r requirements.txt`


## Configuration
All configuration settings are stored in `config.json`, which includes:
- Path to the dataset (`london_weather.csv`)
- Model save path (`temperature_model.pkl`)
- API settings (title, version, host, port)

## Model Training
### Linear Regression Model
To train the linear regression model, run:
`python model_training_regression.py`

This script loads `london_weather.csv`, trains a linear regression model to predict `mean_temp`, and saves the model to `temperature_model.pkl`.

### Random Forest Model
To train the random forest model, run: `python model_training_random.py`

This script loads `london_weather.csv`, trains a random forest model to predict `mean_temp`, and saves the model to `temperature_model.pkl`.

## API Usage
### Without GUI
To run the API without a graphical user interface, execute: `python api.py`

The API will run at the host and port specified [http://localhost:8000/](http://localhost:8000/) in `config.json`. You can access the interactive documentation at [http://localhost:8000/docs](http://localhost:8000/).

### With GUI
To run the API with a browser-based GUI, execute: `python api_ui.py`

Then, open your browser and navigate to the configured host and port [http://127.0.0.1:7860](http://127.0.0.1:7860) to interact with the API endpoints via the GUI.






