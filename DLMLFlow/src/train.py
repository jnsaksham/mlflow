import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

def load_data(dataset_path):
    """Load and preprocess the wine quality dataset."""
    data = pd.read_csv(dataset_path, sep=";")
    
    # Split into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
    
    # Define the input features
    train_x = train_data.drop(columns=['quality']).values
    train_y = train_data['quality'].values
    
    test_x = test_data.drop(columns=['quality']).values
    test_y = test_data['quality'].values
    
    # Create validation set from train set
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42
    )
    
    return train_x, train_y, val_x, val_y, test_x, test_y

def create_model(input_shape, mean, var, params):
    """Create and compile the ANN model."""
    model = keras.Sequential([
        keras.Input(input_shape),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1),
    ])
    
    model.compile(
        optimizer=keras.optimizers.legacy.SGD(  # Using legacy optimizer for M1/M2 Macs
            learning_rate=params["lr"],
            momentum=params["momentum"]
        ),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    
    return model

def train_model(params, epochs, train_x, train_y, val_x, val_y, test_x, test_y):
    """Train the model with given parameters and return evaluation results."""
    # Calculate normalization parameters
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)
    
    # Create and compile model
    model = create_model(train_x.shape[1:], mean, var, params)
    
    # Get model signature for MLflow
    signature = infer_signature(train_x, train_y)
    
    # Train the model with MLflow tracking
    with mlflow.start_run(nested=True):
        model.fit(
            train_x,
            train_y,
            validation_data=(val_x, val_y),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        # Evaluate the model
        eval_result = model.evaluate(test_x, test_y)
        eval_rmse = eval_result[1]
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("rmse", eval_rmse)
        
        # Log the model
        mlflow.tensorflow.log_model(model, "ANN model", signature=signature)
        
        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

def objective(params, epochs, train_x, train_y, val_x, val_y, test_x, test_y):
    """Objective function for hyperparameter optimization."""
    return train_model(
        params=params,
        epochs=epochs,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )

def main():
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("/wine-quality-ann")
    
    # Define hyperparameter search space
    space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(0.1)),
        "momentum": hp.uniform("momentum", 0.0, 1.0),
    }
    
    # Load and preprocess data
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "dataset", "wine-quality.csv")
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(dataset_path)
    
    # Run hyperparameter optimization
    with mlflow.start_run():
        trials = Trials()
        best = fmin(
            fn=lambda params: objective(
                params, 3, train_x, train_y, val_x, val_y, test_x, test_y
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=4,
            trials=trials
        )
        
        # Get best run details
        best_run = sorted(trials.results, key=lambda x: x['loss'])[0]
        
        # Log best parameters and metrics
        mlflow.log_params(best)
        mlflow.log_metric("eval_rmse", best_run['loss'])
        
        # Log best model
        signature = infer_signature(train_x, train_y)
        mlflow.tensorflow.log_model(
            best_run['model'],
            "ANN model",
            signature=signature,
        )
        
        print(f"Best parameters: {best}")
        print(f"Best loss: {best_run['loss']}")

if __name__ == "__main__":
    main()
