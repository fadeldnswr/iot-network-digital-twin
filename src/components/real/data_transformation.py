'''
This module contains the DataTransformation class, which is responsible for
transforming data for machine learning models.
It includes methods for encoding categorical features, scaling numerical features,
'''

import sys
import os
import pandas as pd
import pickle
import numpy as np

from src.exception.exception import CustomException
from src.logging.logging import logging
from src.entity.config_entity import DataTransformationConfig
from src.utils.utils import save_object

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List

# Create data transformation class for real data
class RealDataTransformation:
  '''
  Class to handle data transformation for machine learning models.
  It includes methods for encoding categorical features, scaling numerical features,
  and saving the preprocessor object.
  '''
  def __init__(self, df: pd.DataFrame, time_column: str, target_column: str, config: DataTransformationConfig):
    self.data_tranformation_config = config
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
  
  def get_series(self) -> pd.Series:
    try:
      # Check if the full processed data is available
      if self.full_processed_data is None:
        raise CustomException("Full processed data is not available. Please run get_data_transformer_object() first.")
      # Create a series from the full processed data
      self.series = self.full_processed_data[self.target_column]
      return self.series
    except Exception as e:
      raise CustomException(e, sys)
  
  def initiate_data_transformation(self) -> List[np.ndarray]:
    '''
    Function to initiate data transformation
    based on the features and target variables.
    '''
    try:
      # Split into features and target variable
      logging.info("Initiating data transformation process.")
      X = self.full_processed_data.drop(columns=[self.target_column], axis=1)
      y = self.full_processed_data[self.target_column]
      
      # Concatenate features and target variable
      logging.info("Concatenating features and target variable.")
      full_data = pd.concat([X, y], axis=1)
      full_data = full_data.dropna()
      
      # Define the target column and features
      logging.info("Defining the target column and features.")
      X = full_data.drop(columns=[self.target_column], axis=1)
      y = full_data[self.target_column]
      
      # Scale the features using MinMaxScaler
      features_scaler = MinMaxScaler()
      target_scaler = MinMaxScaler()
      X_scaled = features_scaler.fit_transform(X)
      y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))
      logging.info("Features and target variable have been scaled using MinMaxScaler.")
      
      # Split the data into training and testing sets
      X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False
      )
      logging.info("Data transformation process completed successfully.")
      
      # Save the features scaler
      logging.info("Saving the features and target scalers.")
      save_object(
        file_path=self.data_tranformation_config.features_scaler_file_path,
        obj=features_scaler,
        is_real=True
      )
      # Save the target scaler
      save_object(
        file_path=self.data_tranformation_config.target_scaler_file_path,
        obj=target_scaler,
        is_real=True
      )
      # Return the transformed data
      logging.info("Transformed data has been saved successfully.")
      return [X_train, X_test, y_train, y_test]
    except Exception as e:
      raise CustomException(e, sys)