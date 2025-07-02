'''
User Configuration Schema for IoT Network Digital Twin API
This module defines the schema for user configuration settings in the IoT Network Digital Twin API.
It includes the necessary imports and defines the UserConfig class with its attributes.
'''

from pydantic import BaseModel, Field
from typing import List

class SaveUserConfig(BaseModel):
  '''
  Schema for saving user configuration settings.
  '''
  user_id: str = Field(..., description="Unique identifier for the user", gt=0)
  config_name: str = Field(..., description="Name of the configuration", max_length=20)
  hours: int 
  steps: int
  timestamp: List[str]
  predictions: List[float]