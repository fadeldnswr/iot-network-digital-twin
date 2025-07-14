'''
This module defines the API endpoints for visualizing error metrics related to the IoT network and digital twin
concepts.
'''
import numpy as np

from fastapi import APIRouter, HTTPException, Query
from src.api.db.supabase_config import create_supabase_client
from datetime import datetime
from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error, 
  mean_absolute_percentage_error
)

# Define the API router
router = APIRouter()

# Load real and simulated data from Supabase
supabase = create_supabase_client()

# Define stats summary visualization endpoint
@router.get("/metrics")
async def statistical_summary(
  start_date: datetime = Query(..., description="Start date for data retrieval in ISO format"),
  steps: int = Query(..., description="Steps for data retrieval"),
  category: str = Query(..., description="Category of data to filter by (temperature, humidity(%), latency(ms), rssi(dBm))"),
):
  '''
  Function to visualize error metrics from the IoT network.
  '''
  # Define variable to hold all data
  batch_size = 1000
  offset = 0
  real_data_list = []
  simulated_data_list = []
  
  # Define a safe category name to avoid SQL injection
  safe_category = f'"{category}"'
  field_name = category
  try:
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
    
    # Check the numerical values from the category
    real_values = [item[field_name] for item in real_data_list if isinstance(item[field_name], (int, float))]
    simulated_values = [item[field_name] for item in simulated_data_list if isinstance(item[field_name], (int, float))]
    
    # Create minimum for matching timestamps
    n = min(len(real_values), len(simulated_values))
    real_values = real_values[:n]
    simulated_values = simulated_values[:n]
    
    # Check if the n equals zero
    if n == 0:
      raise HTTPException(status_code=404, detail="No data found for the specified category and date range.")
    
    # Create performance metrics for real and simulated data
    error = {
      "MAE": round(float(mean_absolute_error(real_values, simulated_values)), 2),
      "MSE": round(float(mean_squared_error(real_values, simulated_values)), 2),
      "RMSE": round(float(np.sqrt(mean_squared_error(real_values, simulated_values))), 2),
      "MAPE": round(float(mean_absolute_percentage_error(real_values, simulated_values) * 100), 2)
    }
    
    # Return the performance metrics in a structured format
    return {
      "status": 200,
      "message": f"Performance metrics for {category} data retrieved successfully.",
      "data": error,
      "real_data_count": len(real_values),
      "simulated_data_count": len(simulated_values)
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))