'''
Spline Interpolation Example
This example demonstrates how to use spline interpolation to smooth noisy data.
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from src.services.constants import *
from src.exception.exception import CustomException

# Create Spline Interpolation Class
class SplineInterpolation:
  def __init__(self, smoothing_factor, minutes, amplitude, mean):
    self.smoothing_factor = smoothing_factor
    self.minutes = np.arange(minutes)
    self.amplitude = amplitude
    self.mean = mean
    self.spline = None
    self.smooth_data = None
    self.noisy_data = None
  
  def generate_spline_function(self):
    '''
    Generates a spline function based on the provided parameters.
    '''
    # Generate synthetic sine wave data with noise and trend
    trend = 0.01 * self.minutes
    seasonal = self.amplitude * np.sin(2 * np.pi * self.minutes / 1440)
    noise = 0.75 * np.random.randn(len(self.minutes))
    
    # Combine the components to create noisy data
    self.noisy_data = self.mean + trend + seasonal + noise
    self.spline = InterpolatedUnivariateSpline(self.minutes, self.noisy_data, k=1)
    self.spline.set_smoothing_factor(self.smoothing_factor) # Adjust smoothing factor as needed
    
    # Get smoothed temperature values
    self.smooth_data = self.spline(self.minutes)
    return self.spline, self.smooth_data
  
  def plot_spline(self):
    '''
    Plots the original and smoothed data.
    '''
    # Plot the result
    plt.figure(figsize=(12, 4))
    plt.plot(self.minutes, self.noisy_data, label="Original (Noisy)", alpha=0.5)
    plt.plot(self.minutes, self.smooth_data, label="Spline Smoothed", linewidth=2)
    plt.title("Spline Interpolation (Smoothing)")
    plt.xlabel("Minute")
    plt.ylabel("Data")
    plt.grid(True)
    plt.legend()
    plt.show()


# Check if the script is run directly
if __name__ == "__main__":
  try:
    # Create an instance of SplineInterpolation
    spline_interpolator = SplineInterpolation(
      smoothing_factor=400,  # Adjust smoothing factor as needed
      minutes=1440,          # 24 hours in minutes
      amplitude=AMPLITUDE_HUMIDITY,  # Amplitude of the sine wave
      mean=MEAN_HUMIDITY          # Mean temperature
    )
    # Generate the spline function and smoothed data
    spline_interpolator.generate_spline_function()
    spline_interpolator.plot_spline()
  except Exception as e:
    raise CustomException(e, sys)