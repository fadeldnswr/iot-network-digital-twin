'''
Utility functions for various tasks.
'''

import pandas as pd
import os
import sys
import numpy as np
import dill

from datetime import datetime, timedelta
from src.exception.exception import CustomException
from src.logging.logging import logging
from typing import Tuple, List
from datetime import datetime
from src.api.db.supabase_config import create_supabase_client

# Import metrics for evaluation
from sklearn.metrics import (
  mean_squared_error, mean_absolute_error,
  r2_score, mean_absolute_percentage_error
)
from tensorflow.keras.models import load_model

# Create function to convert simulation data to DataFrame and export to CSV
def save_simulation_data(data, output_dir:str, file_name=None):
  '''
  Saves simulation data to a CSV file.
  Parameters:
  dara (list): List of dictionaries containing simulation data.
  output_dir (str): Directory to save the CSV file.
  file_name (str): Name of the CSV file. If None, uses current date and time.
  '''
  try:
    # Convert packets to rows
    rows = []
    for packet in data:
      timestamp = datetime.combine(datetime.today(), datetime.min.time()) + timedelta(minutes=packet["timestamp"])
      rows.append({
        "timestamp": timestamp.strftime('%Y-%m-%d | %H:%M:%S'),
        "temperature": packet["temperature"],
        "humidity(%)": packet["humidity"],
        "latency(ms)": packet["latency"],
        "throughput(bytes/sec)": packet["throughput"],
        "packet_loss(%)": packet["packet_loss"],
        "rssi(dBm)": packet["rssi"],
      })
    df = pd.DataFrame(rows)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate default file name if not provided
    if file_name is None:
      file_name = f"simulation_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    file_path = os.path.join(output_dir, file_name)
    df.to_csv(file_path, index=False)
    return df, file_path
  except Exception as e:
    raise CustomException(e, sys)

# Define fourier series function
def fourier_series(t, *a):
  ret = a[0] #a0
  N = (len(a) - 1) // 2
  for n in range(1, N + 1):
    ret += a[2 * n-1] * np.cos(2 * np.pi * n * t / 1440) + a[2*n] * np.sin(2 * np.pi * n * t / 1440)
  return ret

# Create save object function
def save_object(file_path, obj, is_real:bool):
  '''
  Save the object to a file using pickle.
  Parameters:
  file_path (str): The path to the file where the object will be saved.
  obj: The object to save.
  '''
  try:
    # Create prefix to the file path based on whether it's real data or not
    prefix = "real_data_" if is_real else "simulated_data_"
    file_name = prefix + file_path
    
    # Set full path
    full_path = os.path.join("artifacts", file_name)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Save the object using dill
    with open(full_path, "wb") as file:
      dill.dump(obj, file)
  except Exception as e:
    raise CustomException(e, sys)

# Create load object function
def load_object(file_path):
  '''
  Load the object from a file using pickle.
  Parameters:
  file_path (str): The path to the file where the object is saved.
  Returns:
  The loaded object.
  '''
  try:
    with open(file_path, "rb") as file:
      return dill.load(file)
  except Exception as e:
    raise CustomException(e, sys)

# Create model evaluation function
def evaluate_model(y_test, y_pred) -> pd.DataFrame:
  '''
  Evaluate the model using various metrics.
  Parameters:
  y_test (array-like): The true values.
  y_pred (array-like): The predicted values.
  Returns:
  A dictionary containing the evaluation metrics.
  '''
  try:
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
      "Metrics": ["MSE", "MAE", "R2", "MAPE", "RMSE"],
      "Values": [mse, mae, r2, mape, rmse]
    })
    return metrics_df
  except Exception as e:
    raise CustomException(e, sys)

# Create function to load LSTM model
def load_lstm_model(model_path : str) -> object:
  '''
  Load the LSTM model from the specified path.
  Parameters:
  model_path (str): The path to the LSTM model file.
  Returns:
  The loaded LSTM model.
  '''
  try:
    if not model_path:
      raise ValueError("Model path cannot be empty.")
    model = load_model(model_path)
    return model
  except Exception as e:
    raise CustomException(e, sys)

# Create data preprocessing function
def data_preprocessing(df: pd.DataFrame, is_real: bool) -> Tuple[List[List[float]], List[str]]:
  '''
  Preprocess the input DataFrame by converting the timestamp to datetime format
  and setting it as the index.
  '''
  try:
    
    # Set the timestamp as datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d | %H:%M:%S")
    df.set_index("timestamp", inplace=True) # Set timestamp as index
    
    # Check if its real data or simulated data
    if is_real:
      # Resample the data to 1-minute intervals and fill missing values
      df = df.resample("min").mean()
      df.fillna(df.mean(), inplace=True)
      df = df.loc["2025-05-12 12:00:00":"2025-05-13 12:00:00"]
    
    data_values = df.select_dtypes(include="number").values.tolist()  # Select numeric columns
    timestamps = df.index.strftime('%Y-%m-%dT%H:%M:%S').tolist()  # Format timestamps
    
    # Select the specified columns and return it as a series
    return data_values, timestamps
  except Exception as e:
    raise CustomException(e, sys)

# Define function to insert data into Supabase
def insert_data_to_supabase(table_name: str, data, batch_size: int = 1000):
  '''
  Insert data into a Supabase table in batches.
  Parameters:
  table_name (str): The name of the Supabase table.
  data (list): The data to insert into the table.
  batch_size (int): The number of records to insert in each batch.
  '''
  try:
    supabase = create_supabase_client()
    for i in range(0, len(data), batch_size):
      batch = data[i:i + batch_size]
      supabase.table(table_name).insert(batch).execute()
    print(f"Inserted {len(data)} records into {table_name} table.")
  except Exception as e:
    raise CustomException(e, sys)

# Define function to format the data from Simpy simulation
def format_data(logged_data, start_time: datetime, seconds: int = 2) -> List[dict]:
  '''
  Format the logged data from the Simpy simulation.
  '''
  try:
    formatted_data = []
    for entry in logged_data:
      formatted_data.append({
        "timestamp": (start_time + timedelta(seconds=int(entry["timestamp"]) * seconds)).isoformat(),
        "temperature": entry["temperature"],
        "humidity(%)": entry["humidity"],
        "latency(ms)": entry["latency"],
        "throughput(bytes/sec)": entry["throughput"],
        "packet_loss(%)": entry["packet_loss"],
        "rssi(dBm)": entry["rssi"],
      })
    return formatted_data
  except Exception as e:
    raise CustomException(e, sys)