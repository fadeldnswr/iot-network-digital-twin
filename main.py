'''
Main script to run the application.
'''

from fastapi import FastAPI, HTTPException
from src.api.routes import (
  real_data,
  error_metrics,
  stats_summary,
  simulated_data,
  user_configuration
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
  title="Internet of Things and Digital Twin API",
  description="API for creating simulation and prediction to Internet of Things data with Digital Twin concepts.",
  version="0.1.0",
  contact={
    "name": "Fadel Achmad Daniswara",
    "email": "daniswarafadel@gmail.com"
  }
)

# Define CORS middleware to allow cross-origin requests
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], 
  allow_credentials=True,
  allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
  allow_headers=["*"],  # Allow all headers
)

# Define the API routers
app.include_router(real_data.router, prefix="/visualize-real", tags=["Real Data"])
app.include_router(simulated_data.router, prefix="/visualize-simulated", tags=["Simulated Data"])
app.include_router(stats_summary.router, prefix="/statistical", tags=["Statistical Summary"])
app.include_router(error_metrics.router, prefix="/error", tags=["Performance Metrics"])

# Define a simple root endpoint
@app.get("/")
async def first_simple_route():
  '''
  Root endpoint to check if the API is running.
  '''
  try:
    return  {
      "message": "Welcome to the Internet of Things and Digital Twin API!",
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))