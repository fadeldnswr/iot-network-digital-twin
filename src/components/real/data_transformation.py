'''
This module contains the DataTransformation class, which is responsible for
transforming data for machine learning models.
It includes methods for encoding categorical features, scaling numerical features,
'''

import sys
import os
import pandas as pd

from src.exception.exception import CustomException
from src.logging.logging import logging
from src.entity.config_entity import DataTransformationConfig

# Create data transformation class for real data
class RealDataTransformation:
  '''
  Class to handle data transformation for machine learning models.
  It includes methods for encoding categorical features, scaling numerical features,
  and saving the preprocessor object.
  '''
  def __init__(self, df: pd.DataFrame, time_column: str, target_column: str):
    self.data_tranformation_config: DataTransformationConfig
    self.time_column = time_column
    self.target_column = target_column
    self.df = df
    self.series = None # Series to store the time series data
    self.full_processed_data = None # DataFrame to store the processed data
  
  def get_data_transformer_object(self) -> pd.DataFrame:
    '''
    The function to get the data transformer object.
    It includes encoding categorical features, scaling numerical features,
    '''
    try:
      # Convert the time column to datetime format
      self.df[self.time_column] = pd.to_datetime(self.df[self.time_column], format="%Y-%m-%d | %H:%M:%S")
      logging.info("Time column has been converted to datetime format.")
      
      # Set the time column as the index
      self.df.set_index(self.time_column, inplace=True)
      logging.info("Time column has been set as the index.")
      
      # Resample the data to 1 minute intervals
      self.df = self.df.resample("min").mean()
      logging.info("Data has been resampled to 1 minute intervals.")
      logging.info(f"Null values in the data: {self.df.isnull().sum()}")
      
      # Imputate missing values using forward fill method
      self.df.fillna(self.df.mean(), inplace=True)
      logging.info("Missing values have been imputed using forward fill method.")
      
      # Show the data for 24 hours
      self.df = self.df.loc["2025-05-12 12:00:00":"2025-05-13 12:00:00"]
      logging.info(f"Dataset shape for 24 hours: {self.df.shape}")
      
      # Create rolling mean and std deviation features
      self.df["rolling_mean_rssi"] = self.df["rssi(dBm)"].rolling(window=5).mean()
      self.df["rolling_std_rssi"] = self.df["rssi(dBm)"].rolling(window=5).std()
      logging.info("Rolling mean and std deviation features have been created.")
      
      # Full processed data
      logging.info("Data transformation process completed successfully.")
      self.full_processed_data = self.df.copy()
    except Exception as e:
      raise CustomException(e, sys)
  
  def save_for_visualization(self):
    try:
      pass
    except Exception as e:
      raise CustomException(e, sys)
  
  def get_series(self):
    try:
      pass
    except Exception as e:
      raise CustomException(e, sys)
  
  def initiate_data_transformation(self):
    '''
    Function to initiate data transformation
    based on the features and target variables.
    '''
    try:
      pass
    except Exception as e:
      raise CustomException(e, sys)