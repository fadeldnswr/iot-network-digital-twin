'''
Utility functions for various tasks.
'''

import pandas as pd
import os
import sys
import numpy as np

from datetime import datetime, timedelta
from src.exception.exception import CustomException
from src.logging.logging import logging

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
