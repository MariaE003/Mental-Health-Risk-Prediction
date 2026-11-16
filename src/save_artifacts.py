
import joblib
import json
from pathlib import Path


def save_artifacts(models_dir, results_dir):

    models_dir = Path(models_dir)
    results_dir = Path(results_dir)

    with open(results_dir / "evaluation_results.json", "r") as file:
        results = json.load(file)

    final_model = joblib.load(
        models_dir / "logistic_regression_best.pkl"
    )

    joblib.dump(final_model, models_dir / "mental_health_model.pkl")

    print("selected model: Logistic Regression")
    print("test Accuracy:", results["logistic_regression"]["test_metrics"]["accuracy"])
    print("test F1:", results["logistic_regression"]["test_metrics"]["f1_macro"])
