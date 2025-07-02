# 🛰️ IoT Network Digital Twin for Monitoring and Prediction
![alt text](image.png)
## Project Description
This project is an end-to-end implementation of a Digital Twin system for IoT network performance, focusing on real-time monitoring, anomaly prediction, and comparative simulation between real and synthetic data. The system collects data from a physical IoT network, applies data preprocessing and validation, and leverages LSTM-based deep learning models to forecast network quality parameters such as RSSI, latency, and packet loss.

## 🔧 Features
- ✅ Real-time data ingestion from Supabase (IoT sensor data)
- 🔁 Preprocessing pipeline with timestamp handling and resampling
- 🧠 LSTM-based model for RSSI prediction (with *.h5 model export)
- 📊 Side-by-side visualization of real vs simulated digital twin data
- ⚙️ API endpoints (FastAPI) for:
    - Manual prediction with input features
    - Batch prediction for N future hours
    - Visualization and user-defined configuration storage
- 💾 Configuration saving to Supabase (user-defined prediction scenarios)
- 🐳 Ready for Docker deployment
- ⚙️ Modular structure for scalability and maintenance

## 📁 Project Structure
├── artifacts/ # Saved models, scalers, and serialized objects
├── dataset/ # Raw and processed data (CSV, JSON, etc.)
├── model/ # Model outputs and evaluation results
├── notebook/ # Jupyter notebooks for validation and exploration
├── src/
│ ├── api/ # FastAPI initialization (app, CORS, main)
│ ├── components/ # Ingestion, transformation, trainer modules
│ ├── db/ # Supabase API config and clients
│ ├── routes/ # API endpoint definitions
│ ├── schema/ # Pydantic request/response models
│ ├── services/ # Core business logic
│ ├── utils/ # Helper functions and serialization tools
│ ├── logging/ # Logging configuration
│ ├── exception/ # Custom exception classes
│ └── entity/ # Config classes for pipeline
├── main.py # Entry point for FastAPI app
├── Dockerfile # Docker configuration (optional)
├── requirements.txt # Python dependencies
└── README.md # Project documentation