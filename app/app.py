import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Student Mental Health Risk Prediction",
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "mental_health_model.pkl"
DATA_PATH = BASE_DIR / "data" / "student_mental_health_clean.csv"
RESULTS_PATH = BASE_DIR / "data" / "results" / "evaluation_results.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

data = pd.read_csv(DATA_PATH)

with open(RESULTS_PATH, "r") as file:
    results = json.load(file)

feature_columns = data.drop(
    columns=["Student_ID", "Mental_Health_Risk"]
).columns.tolist()

classes = list(model.classes_)


st.title("Student Mental Health Risk Prediction")

st.header("Model Performance")

st.write("Performance of the trained classification models.")

# Comparaison des modeles
comparison = []

for name, result in results.items():

    metrics = result["test_metrics"]

    comparison.append({
        "Model": name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision_macro"],
        "Recall": metrics["recall_macro"],
        "F1-score": metrics["f1_macro"]
    })

comparison_df = pd.DataFrame(comparison)

# mertrics
final_metrics = results["logistic_regression"]["test_metrics"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{final_metrics['accuracy']:.2%}")
col2.metric("Precision", f"{final_metrics['precision_macro']:.2%}")
col3.metric("Recall", f"{final_metrics['recall_macro']:.2%}")
col4.metric("F1-score", f"{final_metrics['f1_macro']:.2%}")

# tableau de comparaison

st.subheader("Models Comparison")

st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.2%}",
        "Precision": "{:.2%}",
        "Recall": "{:.2%}",
        "F1-score": "{:.2%}"
    }),
    use_container_width=True
)


# matrice de confusion

st.subheader("Confusion Matrix - Logistic Regression")

matrix = results["logistic_regression"]["confusion_matrix"]

fig, ax = plt.subplots()

ax.imshow(matrix, cmap="Blues")

ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))

ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, matrix[i][j], ha="center", va="center")

st.pyplot(fig)
plt.close(fig)

st.divider()



st.header("Test Prediction")

st.write("Enter student information to predict the risk category.")

with st.form("prediction_form"):

    input_data = {}

    col1, col2 = st.columns(2)

    numeric_columns = data[feature_columns].select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        col for col in feature_columns
        if col not in numeric_columns
    ]

    # champs numeriques
    for i, col in enumerate(numeric_columns):

        with col1 if i % 2 == 0 else col2:

            input_data[col] = st.number_input(
                col.replace("_", " "),
                value=float(data[col].median())
            )

    # champs categoriques
    for i, col in enumerate(categorical_columns):

        options = data[col].dropna().unique().tolist()

        with col1 if i % 2 == 0 else col2:

            input_data[col] = st.selectbox(
                col.replace("_", " "),
                options
            )

    submitted = st.form_submit_button("Predict Risk")

# Prediction

if submitted:

    new_student = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    prediction = model.predict(new_student)[0]

    st.subheader("Prediction Result")

    if prediction == "Low":
        st.success(f"Predicted Risk: {prediction}")

    elif prediction == "Medium":
        st.warning(f"Predicted Risk: {prediction}")

    else:
        st.error(f"Predicted Risk: {prediction}")

    # probabilites

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(new_student)[0]

        probability_df = pd.DataFrame({
            "Risk": classes,
            "Probability": probabilities
        })