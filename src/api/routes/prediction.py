'''
Prediction route for the IoT and Digital Twin application.
This module defines the API endpoint for making predictions based on input data.
'''

from fastapi import APIRouter, HTTPException, Query
from src.api.schema.input_prediction import InputPrediction
from src.utils.utils import load_lstm_model
from dotenv import load_dotenv
from datetime import datetime, timedelta

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
async def predict(data:InputPrediction, hours: int = Query(default=1, description="Predict the data for the next 'n' hours", gt=0, lt=24)):
  try:
    # Convert into minutes
    steps = hours * 60
    prediction = []
    
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
    
    # Create predictions for the specified number of steps
    for _ in range(steps):
      # Scale the input features
      scaled_input = feature_scaler.transform(input_features)
      
      # Reshape the input for the LSTM model
      reshaped_input = scaled_input.reshape((1, 1, scaled_input.shape[1]))
      
      # Scale the input for the model
      scaled_output = model.predict(reshaped_input)
      
      # Inverse transform the scaled output to get the predicted value
      predicted_value = target_scaler.inverse_transform(scaled_output)[0][0]
      prediction.append(predicted_value) # Append the predicted value to the list
      
      # Update the input features for the next prediction
      input_features[0][-3] = predicted_value  # Update the last feature (rssi) with the predicted value
      
      # Generate prediction time
      timestamp = [datetime.now() + timedelta(minutes=i).strftime("%Y-%m-%dT%H:%M:%S") for i in range(steps)]
      
    # Return the prediction results
    return {
      "data": prediction,
      "steps": steps,
      "timestamp": timestamp,
      "hours_predicted": hours
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))