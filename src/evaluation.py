
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_models(processed_dir,models_dir,results_dir):
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)

    results_dir .mkdir(parents=True,exist_ok=True)

    X_val = pd.read_csv(processed_dir / "X_val.csv")
    y_val = pd.read_csv(processed_dir / "y_val.csv").squeeze()

    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()

    model_names = [
        "logistic_regression",
        "svm",
        "random_forest"
    ]

    all_results = {}

    for name in model_names:

        model = joblib.load(models_dir / f"{name}_best.pkl")

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        metrics = {
            "accuracy":accuracy_score(y_test,y_test_pred),
            "precision_macro":precision_score(
                y_test,y_test_pred,average="macro",zero_division=0
            ),
            "recall_macro": recall_score(
                y_test, y_test_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                y_test, y_test_pred, average="macro", zero_division=0
            )
        } 

        report = classification_report(
            y_test,y_test_pred,output_dict=True,zero_division=0
        )
        matrix = confusion_matrix(
            y_test,y_test_pred,labels=model.classes_
        )

        all_results[name] = {
            "validation_accuracy":accuracy_score(y_val, y_val_pred),
            "validation_f1_macro": f1_score(
                y_val, y_val_pred, average="macro"
            ),
            "test_metrics": metrics,
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }

        print(f"\nModel: {name}")
        print("Test metrics:", metrics)
        print(classification_report(
            y_test, y_test_pred, zero_division=0
        ))

    
    with open(results_dir / "evaluation_results.json", "w") as file:
        json.dump(all_results, file, indent=4)

    print("\nEvaluation completed!")

    return all_results