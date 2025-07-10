'''
Spline Interpolation Example
This example demonstrates how to use spline interpolation to smooth noisy data.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

from scipy.interpolate import UnivariateSpline
from src.services.constants import *
from src.exception.exception import CustomException
from dotenv import load_dotenv

# Create Spline Interpolation Class
class SplineInterpolation:
  def __init__(self, smoothing_factor, data):
    self.smoothing_factor = smoothing_factor
    self.data = data
    self.minute = np.arange(len(data))  # Assuming data is indexed by minute
    self.spline = None
    self.smooth_data = None
  
  def generate_spline_function(self):
    '''
    Generates a spline function based on the provided parameters.
    '''
    # Generate spline function based on the data
    self.spline = UnivariateSpline(self.minute, self.data, s=self.smoothing_factor)
    self.smooth_data = self.spline(self.minute)
    return self.spline, self.smooth_data
  
  def plot_spline(self):
    '''
    Plots the original and smoothed data.
    '''
    # Plot the result
    plt.figure(figsize=(12, 4))
    plt.plot(self.minute, self.data, label="Original", alpha=0.5)
    plt.plot(self.minute, self.smooth_data, label="Smoothed", linewidth=2)
    plt.title("Spline Interpolation Modelling for Humidity")
    plt.xlabel("Minute")
    plt.ylabel("Data")
    plt.grid(True)
    plt.legend()
    plt.show()

# Check if the script is run directly
if __name__ == "__main__":
  try:
    # Load the real data
    load_dotenv()
    ESP32_REAL_DATA = os.getenv("ESP32_REAL_DATA_PATH")

    real_esp_data = pd.read_csv(f"{ESP32_REAL_DATA}esp32_1_data.csv")
    humidity_real = real_esp_data["humidity(%)"].values
    
    # Create an instance of SplineInterpolation
    spline_interpolator = SplineInterpolation(
      smoothing_factor=100,  # Adjust smoothing factor as needed
      data=humidity_real,         
    )
    # Generate the spline function and smoothed data
    spline_interpolator.generate_spline_function()
    spline_interpolator.plot_spline()
  except Exception as e:
    raise CustomException(e, sys)