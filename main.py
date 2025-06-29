'''
Main script to run the application.
'''

from fastapi import FastAPI, HTTPException


app = FastAPI(
  title="Internet of Things and Digital Twin API",
  description="API for creating simulation and prediction to Internet of Things data with Digital Twin concepts.",
  version="0.1.0",
  contact={
    "name": "Fadel Achmad Daniswara",
    "email": "daniswarafadel@gmail.com"
  }
)

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