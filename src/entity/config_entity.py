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
    
    # Check if credentials are set
    if not all([self.supabase_url, self.supabase_key, self.supabase_table]):
      raise ValueError("Supabase credentials are not set in the environment variables.")
    
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
  base_dir: str = "artifacts"
  data_type: str = "real"  # Can be 'real' or 'simulation'
  device_id: str = "esp32"  # Device ID for the data source
  
  def __post_init__(self):
    # Create base directory for real of simulated data
    self.sub_dir = os.path.join(self.base_dir, f"{self.data_type}_data_{self.device_id}")
    os.makedirs(self.sub_dir, exist_ok=True)
  
  # Create preprocessor andf scaler file paths based on data type and device ID
  @property
  def preprocessor_obj_file_path(self) -> str:
    return os.path.join(self.sub_dir, f"{self.data_type}_preprocessor.pkl")
  
  @property
  def features_scaler_file_path(self) -> str:
    return os.path.join(self.sub_dir, f"{self.data_type}_features_scaler.pkl")
  
  @property
  def target_scaler_file_path(self) -> str:
    return os.path.join(self.sub_dir, f"{self.data_type}_target_scaler.pkl")
  
  @property
  def transformed_data_file_path(self) -> str:
    return os.path.join(self.sub_dir, f"{self.data_type}_tranformed_data.csv")

# Create model trainer configuration class
@dataclass
class ModelTrainerConfig:
  '''
  Configuration to store the model training artifacts
  '''
  is_real: bool = True  # Flag to indicate if the model is for real data
  @property
  def save_model_path(self) -> str:
    '''
    Returns the path where the trained model will be saved.
    The path is constructed based on whether the data is real or simulated.
    '''
    model_name = "lstm_real.h5" if self.is_real else "lstm_simulation.h5"
    return os.path.join("artifacts", model_name)

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