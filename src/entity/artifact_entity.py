from dataclasses import dataclass

# Data ingestion artifact class to store paths of ingested data files
@dataclass
class DataIngestionArtifact:
  trained_file_path: str
  test_file_path: str

# Data validation artifact class to store paths of validated data files
@dataclass
class DataValidationArtifact:
  validation_status: bool
  valid_train_file_path: str
  valid_test_file_path: str
  invalid_train_file_path: str
  invalid_test_file_path: str
  drift_report_file_path: str

# Data transformation artifact class to store paths of transformed data files
@dataclass
class DataTransformationArtifact:
  transformed_train_file_path: str
  transformed_test_file_path: str
  transformed_obj_file_path: str

# Model trainer artifact class to store paths of trained model files
@dataclass
class ModelTrainerArtifact:
  trained_model_file_path: str
  train_metric_file_path: str
  test_metric_file_path: str

# Metric artifact class to store paths of metric files
@dataclass
class MetricArtifact:
  mse_score: float
  mae_score: float
  r2_score: float
  mape_score: float
  rmse_score: float