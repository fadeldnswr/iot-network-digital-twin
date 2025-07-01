'''
This module handles data ingestion from various sources.
It includes functions to read data from CSV files, JSON files, and databases.
'''

import os
import sys
import pandas as pd

from src.exception.exception import CustomException
from src.logging.logging import logging
from src.entity.config_entity import DataIngestionConfig
from src.components.real.data_transformation import RealDataTransformation
from src.components.real.model_trainer import ModelTrainer
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig
from supabase import create_client, Client

# Create data ingestion class for real data
class RealDataIngestion:
  '''
  Process to ingest data from various sources.
  It includes methods to read data from CSV files, JSON files, and databases.
  '''
  def __init__(self, config: DataIngestionConfig):
    self.config = config
  
  def _read_from_supabase(self) -> pd.DataFrame:
    try:
      supabase: Client = create_client(self.config.supabase_url, self.config.supabase_key) 
      all_data = []
      offset = 0
      limit = 1000
      while True:
        response = supabase.table(self.config.supabase_table).select("*").range(offset, offset + limit - 1).execute()
        data_chunk = response.data
        
        if not data_chunk:
          break
        all_data.extend(data_chunk)
        offset += limit
      dataframe = pd.DataFrame(all_data)
      logging.info(f"Total records fetched from Supabase: {len(dataframe)}")
      return dataframe
    except Exception as e:
      raise CustomException(e, sys)
  
  def initiate_data_ingestion(self) -> pd.DataFrame:
    '''
    Function to initiate data ingestion.
    It reads data from CSV files, JSON files, and databases.
    '''
    try:
      logging.info("Starting data ingestion process.")
      
      # Read data from Supabase
      data = self._read_from_supabase()
      if data.empty:
        logging.warning("No data found in the Supabase table.")
        print("No data found in the Supabase table.")
      
      # Check if the data is dataframe type
      if data is not None and isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        logging.info("Data successfully converted to DataFrame.")
      
      # Return the ingested data
      logging.info("Data ingestion process completed successfully.")
      return data
    except Exception as e:
      raise CustomException(e, sys)

# Testing the pipeline flow
if __name__ == "__main__":
  try:
    logging.info("Starting the data ingestion pipeline.")
    
    # Data Ingestion
    ingestion_config = DataIngestionConfig()
    data_ingestion = RealDataIngestion(config=ingestion_config)
    df = data_ingestion.initiate_data_ingestion()
    logging.info("Data ingestion completed successfully.")
    
    # Data Transformation
    transformation_config = DataTransformationConfig()
    data_transformation = RealDataTransformation(
      df=df,
      time_column="timestamp",
      target_column="rssi(dBm)",
      config=transformation_config
    )
    data_transformation.get_data_transformer_object()
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = data_transformation.initiate_data_transformation()
    logging.info("Data transformation completed successfully.")
    
    # Model Training
    model_trainer_config = ModelTrainerConfig()
    model = ModelTrainer(
      X_train=X_train_scaled,
      X_test=X_test_scaled,
      y_train=y_train_scaled,
      y_test=y_test_scaled,
      target_scaler=data_transformation.target_scaler,
      config=model_trainer_config
    )
    results = model.initiate_model_training()
    logging.info("Model training completed successfully.")
    logging.info(f"Model evaluation results\n{results}")
    logging.info("Data ingestion pipeline completed successfully.")
  except Exception as e:
    raise CustomException(e, sys)