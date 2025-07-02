'''
Supabase configuration module for the IoT Network Digital Twin project.
This module initializes the Supabase client using environment variables for configuration.
'''

from src.exception.exception import CustomException
from supabase import create_client, Client
from dotenv import load_dotenv

import os
import sys

# Load environment variables
load_dotenv()

# Define Supabase URL and Key
SUPABASE_URL = os.getenv("SUPABASE_API_URL")
SUPABASE_KEY = os.getenv("SUPABASE_API_KEY")

# Create Supabase client
def create_supabase_client() -> Client:
  '''
  Create and return a Supabase client instance.
  Returns:
    Client: An instance of the Supabase client.
  '''
  try:
    # Check if the environment variables are set
    if not SUPABASE_URL or not SUPABASE_KEY:
      raise ValueError("Supabase URL or Key is not set in the environment variables.")
    
    # Create and return the Supabase client
    return create_client(SUPABASE_URL, SUPABASE_KEY)
  except Exception as e:
    raise CustomException(e, sys)