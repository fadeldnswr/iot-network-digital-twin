'''
Main script to run the application.
'''

from fastapi import FastAPI, HTTPException
from src.api.routes import prediction, visualization, user_configuration
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
  allow_origins=["*"],  # Allow all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
  allow_headers=["*"],  # Allow all headers
)

# Define the API routers
app.include_router(prediction.router, prefix="/prediction", tags=["Prediction"])
app.include_router(visualization.router, prefix="/visualization", tags=["Visualization"])
app.include_router(user_configuration.router, prefix="/user-config", tags=["User Configuration"])

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