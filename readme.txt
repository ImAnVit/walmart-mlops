Walmart Sales Prediction MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for predicting Walmart's weekly sales. The project covers the entire ML lifecycle, from data preprocessing to model training, deployment, and monitoring, using a combination of popular tools like Apache Airflow, MLflow, FastAPI, and Docker.

Project Overview
Goal:

Predict Walmart's weekly sales based on various features such as:

Store

Holiday Flag

Temperature

Fuel Price

CPI (Consumer Price Index)

Unemployment Rate

Tools & Technologies Used:

Data Preprocessing: Pandas, Scikit-learn

Model Training: RandomForestRegressor, Scikit-learn

Experiment Tracking: MLflow

Model Deployment: FastAPI

Pipeline Automation: Apache Airflow

Containerization: Docker

Monitoring: Logging (Prometheus/Grafana optional)

How It Works

Data Preprocessing:

The Walmart sales dataset is loaded, cleaned, and preprocessed (handling missing data, scaling features, and feature engineering).

Model Training:

A regression model (RandomForestRegressor) is trained to predict sales.

MLflow is used to track experiments, log model parameters, and store models.

Model Deployment:

A FastAPI app is used to serve the trained model as a REST API for real-time predictions.

Pipeline Automation:

Apache Airflow is used to automate data preprocessing, model training, and evaluation in scheduled intervals.

Containerization:

The FastAPI app is containerized using Docker, ensuring the app runs seamlessly across different environments.

Monitoring (Optional):

Basic logging for monitoring predictions and model performance.

Folder Structure
walmart-mlops/
│
├── data/
│   └── walmart.csv          # Dataset
│
├── src/
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   ├── predict.py           # Model prediction script
│
├── api/
│   └── main.py             # FastAPI app for model deployment
│
├── dags/
│   └── training_pipeline.py # Apache Airflow DAG for automation
│
├── models/                  # Saved models and artifacts
│
├── requirements.txt         # Project dependencies
├── Dockerfile               # Dockerfile for containerization
└── docker-compose.yml       # Docker Compose setup for local development
How to Run the Project
1. Clone the repository:
git clone https://github.com/yourusername/walmart-mlops.git
cd walmart-mlops
2. Install dependencies:
pip install -r requirements.txt
3. Run the FastAPI app (for model predictions):
uvicorn api.main:app --reload

Visit http://127.0.0.1:8000/docs to interact with the API and test predictions.

4. Run the Airflow DAG:

Make sure Apache Airflow is set up and running, then trigger the training pipeline:

airflow dags unpause walmart_ml_pipeline

The DAG will automatically preprocess the data and train the model.

5. Dockerize the FastAPI App:

Build and run the FastAPI app with Docker:

docker build -t walmart-mlops .
docker run -p 8000:8000 walmart-mlops
6. Monitoring (Optional):

For real-time model monitoring, integrate Prometheus and Grafana.

Experiment Tracking with MLflow

All experiments and models are tracked using MLflow. To view the experiment dashboard:

mlflow ui

Open http://localhost:5000 to access the MLflow UI and monitor experiments.

Project Enhancements

Feature Engineering: Add more complex features such as seasonality, trend analysis, and external factors.

Hyperparameter Tuning: Use grid search or random search for hyperparameter optimization.

Model Drift Monitoring: Set up tools like Prometheus or Grafana for long-term monitoring of model performance and data drift.

Contributing

Feel free to fork this project, open issues, or submit pull requests for improvements.

License

This project is licensed under the MIT License.