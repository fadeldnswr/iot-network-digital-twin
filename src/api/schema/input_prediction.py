'''
Schema for input data used in the prediction endpoint.
This module defines the InputPrediction class with its attributes.
It includes the necessary imports and defines the InputPrediction class with its attributes.
'''

from pydantic import BaseModel
from typing import List, Optional

class InputPrediction(BaseModel):
  '''
  Model for input data for prediction.
  '''
  temperature: float
  humidity: float
  latency: float
  packet_loss: float
  throughput: float
  rssi: float
  rolling_mean_rssi: float
  rolling_std_rssi: float