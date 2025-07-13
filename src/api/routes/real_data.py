'''
Routes for visualizing data related to the IoT network and digital twin concepts.
This module defines the API endpoints for visualizing real and simulated data from IoT devices.
'''

from fastapi import APIRouter, HTTPException, Query
from src.api.db.supabase_config import create_supabase_client
from datetime import datetime

# Define the API router
router = APIRouter()

# Load real and simulated data from Supabase
supabase = create_supabase_client()

# Define real data visualization endpoint
@router.get("/real")
async def visualize_real_data(
  start_date: datetime = Query(..., description="Start date for data retrieval in ISO format"),
  steps: int = Query(..., description="Steps for data retrieval"),
  category: str = Query(..., description="Category of data to filter by (temperature, humidity(%), latency(ms), rssi(dBm))"),
):
  '''
  Function to visualize real data from the IoT network.
  '''
  # Define the offset and batch size for pagination
  offset = 0
  batch_size = 1000
  all_data = []
  
  # Define a safe category name to avoid SQL injection
  safe_category = f'"{category}"'
  try:
    while offset < steps:
      range_end = min(offset + batch_size - 1, steps - 1)
      # Fetch data in batchs to avoid memory issues with large datasets
      response = supabase.table("esp32_real_data").select(f'{safe_category}, timestamp') \
        .gte("timestamp", start_date) \
        .order("timestamp", desc=False) \
        .range(offset, range_end) \
        .execute()
      
      # Check if the response contains data
      batch = response.data
      if not batch:
        break
      
      # Preprocess the data if necessary
      all_data.extend(batch)
      offset += batch_size
    
    # Return the data in a structured format
    return {
      "status": 200,
      "message": f"{category} data retrieved successfully.",
      "data": all_data,
      "count": len(all_data)
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))