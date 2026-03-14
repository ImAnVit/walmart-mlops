from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.preprocess import preprocess_data
from src.train import train

default_args = {
    "owner": "mlops",
    "start_date": datetime(2024, 1, 1),
}

dag = DAG(
    "walmart_ml_pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False
)

task1 = PythonOperator(
    task_id="preprocess",
    python_callable=preprocess_data,
    dag=dag
)

task2 = PythonOperator(
    task_id="train_model",
    python_callable=train,
    dag=dag
)

task1 >> task2