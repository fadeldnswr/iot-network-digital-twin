'''
Configuration entity for data ingestion.
This module defines the configuration for data ingestion, including paths for raw and processed data.
'''
import os, sys
from src.services.constants import *
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

class DataIngestionConfig:
  '''
  Configuration class for data ingestion.
  It defines the paths for raw data and processed data.
  '''
  def __init__(self):
    # Define supabase configuration
    self.supabase_url: str = os.getenv("SUPABASE_API_URL")
    self.supabase_key: str = os.getenv("SUPABASE_API_KEY")
    self.supabase_table: str = os.getenv("SUPABASE_TABLE")
    
    # Define paths for raw and processed data
    PROJECT_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(PROJECT_ROOT, "dataset")
    self.raw_data_path: str = os.path.join(data_dir, "raw_data.csv")
    self.train_data_path: str = os.path.join(data_dir, "train_data.csv")
    self.test_data_path: str = os.path.join(data_dir, "test_data.csv")
    self.source_path: str = os.path.join(PROJECT_ROOT, "dataset" ,"esp32_1_data.csv")

# Define data transformation configuration class
@dataclass
class DataTransformationConfig:
  '''
  Configuration class for data transformation.
  It defines the paths for transformed data and the model directory.
  '''
  @property
  def preprocessor_obj_file_path(self):
    return os.path.join("artifacts", f"esp32_data_preprocessor.pkl")

# Define data transformation configuration for simulation data
@dataclass
class SimulationDataTransformationConfig:
  '''
  Configuration class for simulation data transformation.
  It defines the paths for transformed data and the model directory.
  '''
  @property
  def preprocessor_obj_file_path(self):
    return os.path.join("artifacts", f"simulation_data_preprocessor.pkl")