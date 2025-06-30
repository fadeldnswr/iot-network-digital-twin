'''
This module handles data ingestion from various sources.
It includes functions to read data from CSV files, JSON files, and databases.
'''

import os
import sys
import pandas as pd

from src.exception.exception import CustomException
from logging.logging import logging
from src.entity.config_entity import DataIngestionConfig
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
      response = supabase.table(self.config.supabase_table).select("*").execute()
      dataframe = pd.DataFrame(response.data)
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