'''
This module contains the DataTransformation class, which is responsible for
transforming data for machine learning models.
It includes methods for encoding categorical features, scaling numerical features,
'''

import sys
import os
import pandas as pd

from dataclasses import dataclass
from src.exception.exception import CustomException
from src.logging.logging import logging

@dataclass
class DataTransformationConfig:
  '''
  Configuration class for data transformation.
  It defines the paths for transformed data and the model directory.
  '''
  device_id: str
  @property
  def preprocessor_obj_file_path(self):
    return os.path.join("artifacts", f"{self.device_id}_preprocessor.pkl")

# Create data transformation class
class DataTransformation:
  '''
  Class to handle data transformation for machine learning models.
  It includes methods for encoding categorical features, scaling numerical features,
  and saving the preprocessor object.
  '''
  def __init__(self):
    pass
  
  def get_data_transformer_object(self):
    '''
    The function to get the data transformer object.
    It includes encoding categorical features, scaling numerical features,
    '''
    try:
      pass
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