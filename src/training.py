
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def train_models(processed_dir,preprocessign_path,model_dir):
    processed_dir = Path(processed_dir)
    model_dir = Path(model_dir)

    model_dir.mkdir(parents=True,exist_ok=True)

    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()

    preprocessing = joblib.load(preprocessign_path)

    model = { 
        "logistic_regression":{
            "model":LogisticRegression(max_iter=1000),
            "params":{
                "model__C":[0.01, 0.1, 1, 10, 100]
            }
        },
        "svm":{
            "model": SVC(),
            "params":{
                "model__C":[0.1,1,10,100],
                "model__kernel":["linear","rbf"],
                "model__gamma":["scale","auto"],
            }
        },
        "random_forest":{
            "model" : RandomForestClassifier(random_state=42),
            "params":{
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        }
    } 

    result ={}

    for name,config in model.items():
        piplinne = Pipeline([
            ("preprocessing",preprocessing),
            ("model",config["model"])
        ])
        grid = GridSearchCV(
            estimator=piplinne,
            param_grid=config["params"],
            cv=5,
            scoring="f1_macro",
            n_jobs=-1
        )

        grid.fit(X_train,y_train)

        best_model = grid.best_estimator_
        joblib.dump(best_model,model_dir / f"{name}_best.pkl")

        result[name] = {
            "best_params": grid.best_params_,
            "best_cv_score": grid.best_score_
        }

        print(f"\nModel: {name}")
        print("Best parameters:", grid.best_params_)
        print("Best CV score:", grid.best_score_)

    return result