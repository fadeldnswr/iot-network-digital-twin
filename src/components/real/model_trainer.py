'''
This module contains the ARIMA model training class,
which is responsible for training the ARIMA model
on the time series data.
'''

import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.exception.exception import CustomException
from src.logging.logging import logging
from src.entity.config_entity import ModelTrainerConfig
from src.components.real.data_transformation import RealDataTransformation
from src.utils.utils import evaluate_model
from sklearn.preprocessing import MinMaxScaler

# Import metrics for evaluation
from sklearn.metrics import (
  mean_squared_error, mean_absolute_error,
  r2_score, mean_absolute_percentage_error
)

# Import LSTM model
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Create model trainer class
class ModelTrainer:
  '''
  Model trainer class to handle the training of LSTM models.
  It includes methods for training the model, evaluating it,
  '''
  def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, target_scaler: MinMaxScaler, config: ModelTrainerConfig) -> pd.DataFrame:
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.target_scaler = target_scaler
    self.model_config = config
  
  def reshape(self, X):
    '''
    Reshape the input data to 3D shape for LSTM model.
    The input data should be in the shape of (samples, timesteps, features).
    '''
    if len(X.shape) == 2:
      # If the input is 2D, reshape it to 3D
      return X.reshape((X.shape[0], 1, X.shape[1]))
  
  def initiate_model_training(self):
    '''
    Initiates the model training process.
    It builds the LSTM model, trains it on the training data,
    evaluates it on the test data, and saves the trained model.
    '''
    try:
      logging.info("Initiating model training process.")
    # Use sequences directly; shape = (samples, timesteps, features)
      X_train_lstm = self.reshape(self.X_train)
      X_test_lstm = self.reshape(self.X_test)

      # Build model
      timesteps = X_train_lstm.shape[1]
      n_features = X_train_lstm.shape[2]
      # Reshape the input data for LSTM model
      logging.info("Reshaping the input data for LSTM model.")
      
      # Model building
      logging.info("Building the LSTM model.")
      model = Sequential([
        LSTM(units=64, return_sequences=True, input_shape=(timesteps, n_features)),
        Dropout(0.2),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(1)  # Output layer for regression
      ])
      # Compile the model
      model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_absolute_error", metrics=["mae"])
      model.summary()
      
      # Train the model
      history = model.fit(
        X_train_lstm, self.y_train,
        validation_data=(X_test_lstm, self.y_test),
        epochs=100,
        batch_size=32,
        verbose=1
      )
      logging.info("Model training completed successfully.")
      
      # Evaluate the model
      logging.info("Evaluating the model on test data.")
      y_pred_scaled = model.predict(X_test_lstm)
      
      # Inverse transform the predictions
      y_pred = self.target_scaler.inverse_transform(y_pred_scaled)
      y_test = self.target_scaler.inverse_transform(self.y_test)
      model_evaluation = evaluate_model(y_test, y_pred)
      logging.info(f"Model evaluation metrics: {model_evaluation}")
      
      # Save the trained model
      logging.info("Saving the trained model.")
      model_path = self.model_config.save_model_path
      os.makedirs(os.path.dirname(model_path), exist_ok=True)
      save_model(model, model_path)
      logging.info(f"Model saved at {model_path}")
      
      return model_evaluation
    except Exception as e:
      raise CustomException(e, sys)