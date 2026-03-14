import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train():

    # Load preprocessed data
    X_train, X_test, y_train, y_test = joblib.load("data/train_test.pkl")

    # Set the experiment name in MLflow
    mlflow.set_experiment("walmart_sales")

    # Start an MLflow run to track this experiment
    with mlflow.start_run():

        # Define the model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Log model parameters (hyperparameters)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Calculate evaluation metrics (MSE and RMSE)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5

        # Log metrics to MLflow
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)

        # Log the trained model
        mlflow.sklearn.log_model(model, "model")

        # Optionally, save the model to disk as well
        joblib.dump(model, "models/model.pkl")

        # Log additional experiment details (optional)
        mlflow.log_param("data_version", "v1")  # Example of a custom parameter

if __name__ == "__main__":
    train()