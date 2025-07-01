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
from typing import List, Tuple

# Create data transformation class for real data
class RealDataTransformation:
  '''
  Class to handle data transformation for machine learning models.
  It includes methods for encoding categorical features, scaling numerical features,
  and saving the preprocessor object.
  '''
  def __init__(self, df: pd.DataFrame, time_column: str, config: DataTransformationConfig, target_column: str, window_size: int = 5):
    self.data_tranformation_config = config
    self.time_column = time_column
    self.target_column = target_column
    self.df = df.copy()
    self.config = config # Configuration for data transformation
    self.window_size = window_size # Window size for creating sequences
    self.series = None # Series to store the time series data
    self.full_processed_data = None # DataFrame to store the processed data
    self.target_scaler: MinMaxScaler = None # Scaler for the target variable
    self.feature_scaler: MinMaxScaler = None # Scaler for the features
  
  def get_data_transformer_object(self) -> pd.DataFrame:
    '''
    The function to get the data transformer object.
    It includes encoding categorical features, scaling numerical features,
    '''
    try:
      # Convert the time column to datetime format
      self.df[self.time_column] = pd.to_datetime(self.df[self.time_column], format="%Y-%m-%dT%H:%M:%S")
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
      self.df["rolling_mean_rssi"] = self.df["rssi(dBm)"].rolling(window=5, min_periods=1).mean()
      self.df["rolling_std_rssi"] = self.df["rssi(dBm)"].rolling(window=5, min_periods=1).std()
    
      logging.info("Rolling mean and std deviation features have been created.")
      
      # Full processed data
      logging.info(f"Data transformation process completed successfully with shape {self.df.shape}.")
      self.full_processed_data = self.df.copy()
    except Exception as e:
      raise CustomException(e, sys)
  
  def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create sequences for LSTM model.
    Parameters:
    X (np.ndarray): Input features.
    y (np.ndarray): Target variable.
    Returns:
    Tuple[np.ndarray, np.ndarray]: Sequences of input features and target variable.
    '''
    Xs, ys = [], []
    for i in range(len(X) - self.window_size):
      Xs.append(X[i : i + self.window_size])
      ys.append(y[i + self.window_size])
    return np.array(Xs), np.array(ys)
  
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
      
      # Combine features and target into one DataFrame, then drop NaNs
      df = self.full_processed_data.copy()
      X_df = df.drop(columns=[self.target_column])
      y_df = df[[self.target_column]]
      full_df = pd.concat([X_df, y_df], axis=1).dropna()

      # Prepare arrays
      X = full_df.drop(columns=[self.target_column]).values
      y = full_df[self.target_column].values.reshape(-1, 1)

      # Scale features and target
      self.feature_scaler = MinMaxScaler()
      self.target_scaler = MinMaxScaler()
      X_scaled = self.feature_scaler.fit_transform(X)
      y_scaled = self.target_scaler.fit_transform(y)

      # Split into training and testing sets without shuffling
      X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=0.2,
        shuffle=False
      )
      
      # Save the features scaler
      logging.info("Saving the features and target scalers.")
      save_object(
        file_path=self.data_tranformation_config.features_scaler_file_path,
        obj=self.feature_scaler,
        is_real=True
      )
      # Save the target scaler
      save_object(
        file_path=self.data_tranformation_config.target_scaler_file_path,
        obj=self.target_scaler,
        is_real=True
      )
      # Return the transformed data
      logging.info("Transformed data has been saved successfully.")
      return [X_train, X_test, y_train, y_test]
    except Exception as e:
      raise CustomException(e, sys)
  
  def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
    '''
    Function to inverse transform the scaled target variable.
    Parameters:
    scaled_values (np.ndarray): Scaled target variable values.
    Returns:
    np.ndarray: Inverse transformed target variable values.
    '''
    try:
      if self.target_scaler is None:
        raise CustomException("Target scaler is not available. Please run initiate_data_transformation() first.")
      return self.target_scaler.inverse_transform(scaled_values)
    except Exception as e:
      raise CustomException(e, sys)