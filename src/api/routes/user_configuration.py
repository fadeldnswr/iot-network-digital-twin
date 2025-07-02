'''
User coinfiguration routes for managing user settings and preferences.
This module defines the API endpoints for user configuration, allowing users to manage their settings
and preferences within the IoT Network Digital Twin application.
'''

from fastapi import APIRouter, HTTPException
from src.api.schema.user_config import SaveUserConfig
from src.api.db.supabase_config import create_supabase_client

# Define the API router
router = APIRouter()

# Define the Supabase client
supabase = create_supabase_client()

# Create a new user configuration routes
@router.post("/save-configuration")
async def save_user_configuration(data: SaveUserConfig):
  try:
    # Create a new user configuration in the database
    response = supabase.table("user_configuration").insert({
      "user_id": data.user_id,
      "config_name": data.config_name,
      "hours": data.hours,
      "steps": data.steps,
      "timestamp": data.timestamp,
      "predictions": data.predictions
    }).execute()
    
    # Check is the response is successful
    if response.status_code == 201:
      return {
        "message": "User configuration saved successfully."
      }
    else:
      raise HTTPException(status_code=response.status_code, detail="Failed to save user configuration.")
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))