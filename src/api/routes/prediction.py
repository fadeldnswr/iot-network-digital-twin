'''
Prediction route for the IoT and Digital Twin application.
This module defines the API endpoint for making predictions based on input data.
'''

from fastapi import APIRouter, HTTPException
from src.api.schema.input_prediction import InputPrediction
from src.utils.utils import load_lstm_model
from dotenv import load_dotenv

import numpy as np
import pickle
import os

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_FILE_PATH")
FEATURE_TARGET_PATH = os.getenv("FEATURE_TARGET_PATH")

# Define the API router
router = APIRouter()

# Load the LSTM model
model = load_lstm_model(f"{MODEL_PATH}/lstm_real.h5")

# Load the feature scaler
with open(f"{FEATURE_TARGET_PATH}/real_features_scaler.pkl", "rb") as file:
  feature_scaler = pickle.load(file)

# Load the target scaler
with open(f"{FEATURE_TARGET_PATH}/real_target_scaler.pkl", "rb") as file:
  target_scaler = pickle.load(file)

# Define manual prediction endpoint
@router.post("/")
async def predict(data:InputPrediction):
  try:
    # Connvert input featurs into numpy array
    input_features = np.array([[
      data.temperature,
      data.humidity,
      data.latency,
      data.packet_loss,
      data.throughput,
      data.rssi,
      data.rolling_mean_rssi,
      data.rolling_std_rssi
    ]])
    # Scale the input features
    scaler = feature_scaler.transform(input_features)
    # Reshape input features for the model
    reshaped_features = scaler.reshape((1, 1, scaler.shape[1]))
    # Scaled predictions
    scaled_prediction = model.predict(reshaped_features)
    # Inverse transform the scaled prediction to get the actual value
    prediction = target_scaler.inverse_transform(scaled_prediction)
    # Return the prediction as a float
    return {
      "prediction": float(prediction[0][0])
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))