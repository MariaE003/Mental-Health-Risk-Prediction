
import sys
from datetime import datetime

from airflow.sdk import DAG, task

sys.path.insert(0, "/opt/airflow/src")

from data_preparation import prepare_data
from training import train_models
from evaluation import evaluate_models
from save_artifacts import save_artifacts


DATA_PATH = "/opt/mental-health-risk-prediction/data/student_mental_health_clean.csv"
PROCESSED_DIR = "/opt/mental-health-risk-prediction/data/processed"
MODELS_DIR = "/opt/mental-health-risk-prediction/models"
PREPROCESSING_PATH = "/opt/mental-health-risk-prediction/models/preprocessing.joblib"
RESULTS_DIR = "/opt/mental-health-risk-prediction/data/results"


with DAG(
    dag_id="mental_health_training",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mental-health", "training"]
) as dag:

    @task
    def prepare():
        prepare_data(DATA_PATH, PROCESSED_DIR)
        return True

    @task
    def train(prepared):
        train_models(
            PROCESSED_DIR,
            PREPROCESSING_PATH,
            MODELS_DIR
        )
        return True

    @task
    def evaluate(trained):
        evaluate_models(
            PROCESSED_DIR,
            MODELS_DIR,
            RESULTS_DIR
        )
        return True

    @task
    def save(evaluated):
        save_artifacts(MODELS_DIR, RESULTS_DIR)

    prepared = prepare()
    trained = train(prepared)
    evaluated = evaluate(trained)
    save(evaluated)
