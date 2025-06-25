import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import curve_fit
from src.services.constants import *
from src.utils.utils import fourier_series
from src.exception.exception import CustomException


# Create Fourier Series class
class FourierSeries:
  def __init__(self, N, minutes, amplitude, mean):
    self.N = N
    self.minutes = np.arange(minutes)
    self.amplitude = amplitude
    self.mean = mean
    self.popt = None
    self.noisy_data = None
    self.smooth_data = None
  
  def generate_fourier_series(self):
    '''
    Generates a Fourier series function based on the provided parameters.
    '''
    # Generate synthetic sine wave data with noise
    self.noisy_data = self.mean + self.amplitude * np.sin(2 * np.pi * self.minutes / 1440) + 0.3 * np.random.randn(len(self.minutes))
    self.popt,_ = curve_fit(lambda t, *a:fourier_series(t, *a), self.minutes, self.noisy_data, p0=[0] * (2*self.N+1))
    self.smooth_data = fourier_series(self.minutes, *self.popt)
    
    # Return the fitted parameters and smoothed data
    return self.noisy_data, self.smooth_data

  def plot_fourier_series(self):
    '''
    Plots the original and smoothed data.
    '''
    plt.figure(figsize=(12, 6))
    plt.plot(self.minutes, self.noisy_data, label="Original", alpha=0.5)
    plt.plot(self.minutes, self.smooth_data, label=f"Fourier Series (N={self.N})", linewidth=2)
    plt.title("Fourier Series")
    plt.xlabel("Minute")
    plt.ylabel("Data")
    plt.grid(True)
    plt.legend()
    plt.show()

# Check if the script is run directly
if __name__ == "__main__":
  try:
    # Create an instance of SplineInterpolation
    fourier_series_eq = FourierSeries(
      N=10,                # Number of harmonics
      minutes=1440,       # 24 hours in minutes
      amplitude=4.152174e+00,  # Amplitude of the sine wave
      mean=MEAN_TEMP       # Mean temperature
    )
    # Generate the spline function and smoothed data
    fourier_series_eq.generate_fourier_series()
    fourier_series_eq.plot_fourier_series()
  except Exception as e:
    raise CustomException(e, sys)