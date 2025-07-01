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