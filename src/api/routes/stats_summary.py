'''
Routes for visualizing stats summary related to the IoT network and digital twin concepts.
This module defines the API endpoints for visualizing stats summary from IoT devices.
'''

import numpy as np

from fastapi import APIRouter, HTTPException, Query
from src.api.db.supabase_config import create_supabase_client
from datetime import datetime

# Define the API router
router = APIRouter()

# Load real and simulated data from Supabase
supabase = create_supabase_client()

# Define stats summary visualization endpoint
@router.get("/summary")
async def statistical_summary(
  start_date: datetime = Query(..., description="Start date for data retrieval in ISO format"),
  steps: int = Query(..., description="End date for data retrieval in ISO format"),
  category: str = Query(..., description="Category of data to filter by (temperature, humidity(%), latency(ms), rssi(dBm))"),
):
  '''
  Function to visualize statistical summary from the IoT network.\
  '''
  try:
    # Define variable to hold all data
    batch_size = 1000
    offset = 0
    real_data_list = []
    simulated_data_list = []
    
    # Define a safe category name to avoid SQL injection
    safe_category = f'"{category}"'
    field_name = category
    while offset < steps:
      range_end = min(offset + batch_size - 1, steps - 1)
      # Fetch real data from supabase
      real_response = supabase.table("esp32_real_data").select(f'{safe_category}, timestamp') \
        .gte("timestamp", start_date) \
        .order("timestamp", desc=False) \
        .range(offset, range_end) \
        .execute()
        
      # Fetch simulated data from supabase
      simulated_response = supabase.table("iot_simulation").select(f'{safe_category}, timestamp') \
        .gte("timestamp", start_date) \
        .order("timestamp", desc=False) \
        .range(offset, range_end) \
        .execute()
        
        # Check if the responses contain data
      real_data = real_response.data
      simulated_data = simulated_response.data
      if not real_data and not simulated_data:
        break
        
      # Preprocess the data if necessary
      real_data_list.extend(real_data)
      simulated_data_list.extend(simulated_data)
      offset += batch_size
      
    # Get numerical values from the category
    real_value = [item[field_name] for item in real_data_list if isinstance(item[field_name], (int, float))]
    simulated_value = [item[field_name] for item in simulated_data_list if isinstance(item[field_name], (int, float))]
    
    # Get minimum for matching timestamps
    n = min(len(real_value), len(simulated_value))
    real_value = real_value[:n]
    simulated_value = simulated_value[:n]
    
    # Create statistics for real data
    real_statistics = {
      "mean": round(float(np.mean(real_value)), 2),
      "median": round(float(np.median(real_value)), 2),
      "std_dev": round(float(np.std(real_value)), 2),
    }
    
    # Create statistics for simulated data
    simulated_statistics = {
      "mean": round(float(np.mean(simulated_value)), 2),
      "median": round(float(np.median(simulated_value)), 2),
      "std_dev": round(float(np.std(simulated_value)), 2),
    }
    
    return {
      "status": 200,
      "message": f"Statistical summary for {category} data retrieved successfully.",
      "real_data_statistics": real_statistics,
      "simulated_data_statistics": simulated_statistics,
      "real_data_count": len(real_data_list),
      "simulated_data_count": len(simulated_data_list),
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))