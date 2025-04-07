import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import os

# Use environment variable if set, otherwise use default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestRegressor()
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def main(test_size, random_state):
    # Set tracking URI to MLflow server
    mlflow.set_tracking_uri("http://localhost:5001")
    
    # Set the experiment name
    mlflow.set_experiment("house_price_prediction")
    
    # Load and prepare data
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['Price'] = housing.target
    
    # Split data
    X = data.drop('Price', axis=1)
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # MLFlow experiment
    with mlflow.start_run() as run:
        # Hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        best_model = grid_search.best_estimator_
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mse", mse)
        
        # Log model
        signature = mlflow.models.infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            registered_model_name="best_model"
        )
        
        print(f"Best hyperparameters: {grid_search.best_params_}")
        print(f"Best MSE: {mse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    
    main(args.test_size, args.random_state) 