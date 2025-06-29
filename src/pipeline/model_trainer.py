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

# Import metrics for evaluation
from sklearn.metrics import (
  mean_squared_error, mean_absolute_error,
  r2_score, mean_absolute_percentage_error
)

# Import LSTM model
from tensorflow.keras.models import Sequntial
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.optimizer import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model

# Create model trainer configurations
@dataclass
class ModelTrainerConfig:
  '''
  Configuration to store the model training artifacts
  '''
  def __init__(self, model_name:str): 
    # Change the model name to a correct one
    self.trained_model_file_path = save_model(model_name, os.path.join("artifacts", f"{model_name}_model.h5"))

# Create model trainer class
class ModelTrainer:
  '''
  Class to train LSTM model on time series data.
  '''
  def __init__(self):
    pass
  
  def initiate_model_training(self):
    try:
      pass
    except Exception as e:
      raise CustomException(e, sys)