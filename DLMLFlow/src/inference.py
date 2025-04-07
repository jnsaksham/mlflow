import os
import logging
import numpy as np
import pandas as pd
import mlflow
from typing import Union, List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WineQualityPredictor:
    def __init__(self, model_uri: str = None, tracking_uri: str = None):
        """
        Initialize the predictor with a model URI and tracking URI.
        
        Args:
            model_uri (str): URI of the model to load (e.g., 'models:/finalmodel/2' or 'runs:/<run_id>/model')
            tracking_uri (str): URI of the MLflow tracking server
        """
        self.model = None
        self.model_uri = model_uri
        
        if tracking_uri is None:
            # Default to local mlruns directory
            tracking_uri = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "mlruns"
            )
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Set MLflow tracking URI to: {tracking_uri}")
    
    def load_model(self, model_uri: str = None) -> None:
        """
        Load the MLflow model.
        
        Args:
            model_uri (str, optional): URI of the model to load. If not provided, uses the one from initialization.
        
        Raises:
            ValueError: If no model URI is provided
            mlflow.exceptions.MlflowException: If model loading fails
        """
        try:
            if model_uri is not None:
                self.model_uri = model_uri
            
            if self.model_uri is None:
                raise ValueError("No model URI provided. Please provide a model URI.")
            
            logger.info(f"Loading model from: {self.model_uri}")
            self.model = mlflow.pyfunc.load_model(self.model_uri)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, data: Union[pd.DataFrame, np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Preprocess the input data into the format expected by the model.
        
        Args:
            data: Input data in various formats
            
        Returns:
            np.ndarray: Preprocessed data ready for prediction
            
        Raises:
            ValueError: If input data format is not supported
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Ensure correct column order if DataFrame
                expected_columns = [
                    'fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
                ]
                if not all(col in data.columns for col in expected_columns):
                    raise ValueError(f"Input DataFrame must contain all required columns: {expected_columns}")
                return data[expected_columns].values
                
            elif isinstance(data, np.ndarray):
                if data.shape[1] != 11:  # Expected number of features
                    raise ValueError("Input array must have 11 features")
                return data
                
            elif isinstance(data, list):
                arr = np.array(data)
                if arr.shape[1] != 11:  # Expected number of features
                    raise ValueError("Input list must have 11 features per sample")
                return arr
                
            else:
                raise ValueError("Unsupported input type. Please provide DataFrame, numpy array, or list of lists")
                
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray, List[List[float]]]) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If input data is invalid
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input data
            processed_data = self.preprocess_input(data)
            
            # Make predictions
            logger.info(f"Making predictions for {len(processed_data)} samples")
            predictions = self.model.predict(processed_data)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

def main():
    """
    Example usage of the WineQualityPredictor class.
    """
    try:
        # Initialize predictor
        predictor = WineQualityPredictor(model_uri='models:/finalmodel/2')
        
        # Load the model
        predictor.load_model()
        
        # Example input data
        sample_data = pd.DataFrame({
            'fixed acidity': [7.0],
            'volatile acidity': [0.27],
            'citric acid': [0.36],
            'residual sugar': [20.7],
            'chlorides': [0.045],
            'free sulfur dioxide': [45.0],
            'total sulfur dioxide': [170.0],
            'density': [1.001],
            'pH': [3.0],
            'sulphates': [0.45],
            'alcohol': [8.8]
        })
        
        # Make predictions
        predictions = predictor.predict(sample_data)
        logger.info(f"Predictions: {predictions}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 