# Student Mental Health Risk Prediction

### 1. Présentation

MindCare AI est un projet de Data Science et d'Intelligence Artificielle dont l'objectif est d'analyser des données relatives aux étudiants et de développer un modèle de classification supervisée permettant d'estimer leur catégorie de risque en matière de santé mentale.

Ce projet met en œuvre un pipeline complet de Machine Learning, depuis l'exploration des données jusqu'au déploiement d'une application interactive.


### 2. Objectifs

Les principaux objectifs du projet sont :

- Explorer et comprendre les données relatives aux étudiants.
- Identifier les facteurs associés aux différentes catégories de risque.
- Préparer les données pour l'apprentissage automatique.
- Entraîner et comparer plusieurs modèles de classification.
- Optimiser les hyperparamètres et évaluer les performances.
- Automatiser le pipeline avec Apache Airflow.
- Développer une application interactive avec Streamlit.

### 3. Source des données

Le projet utilise le dataset suivant :

**Student Mental Health Risk Prediction Dataset**

Source : https://www.kaggle.com/datasets/maqdadali/mental-health-risk-prediction-dataset-students

Caractéristiques du dataset :
- Nombre d'observations : 2 000.
- Nombre de colonnes : 17.
- Variable cible : `Mental_Health_Risk`.
- Classes : `Low`, `Medium` et `High`.
- Identifiant : `Student_ID`, exclu de l'entraînement.

### 4. Environnement technique

| Catégorie | Technologies |
|---|---|
| Langage | Python |
| Manipulation des données | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Préprocessing | Scikit-learn Pipeline, ColumnTransformer |
| Optimisation | GridSearchCV |
| Sauvegarde des modèles | Joblib |
| Environnement expérimental | Jupyter Notebook |
| Orchestration | Apache Airflow |
| Conteneurisation | Docker, Docker Compose |
| Interface utilisateur | Streamlit |
| Versionnement | Git, GitHub |

### 5. Démarche de réalisation

#### 5.1. Analyse exploratoire des données

Cette étape consiste à comprendre la structure et la qualité du dataset.

Travaux réalisés :
- Analyse des dimensions et des types de données.
- Identification des valeurs manquantes.
- Vérification des doublons et des incohérences.
- Détection des valeurs aberrantes avec la méthode IQR.
- Analyse statistique descriptive.
- Étude de la distribution de la variable cible.
- Analyse des corrélations et des relations entre les variables.

#### 5.2. Préparation des données

Les données sont préparées avant leur utilisation dans les modèles.

Les opérations comprennent :
- Traitement des valeurs manquantes.
- Encodage des variables catégorielles.
- Standardisation des variables numériques.
- Séparation des données en ensembles d'entraînement, de validation et de test.
- Intégration des transformations dans un pipeline afin de limiter le Data Leakage.

Répartition utilisée :

| Ensemble | Proportion |
|---|---:|
| Entraînement | 70 % |
| Validation | 15 % |
| Test | 15 % |

La séparation est stratifiée afin de conserver la distribution des classes.

#### 5.3. Entraînement des modèles

Trois algorithmes de classification ont été étudiés :

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest

Les modèles utilisent les mêmes ensembles de données afin de permettre une comparaison cohérente.

#### 5.4. Optimisation et évaluation

L'optimisation des hyperparamètres est effectuée avec GridSearchCV et une validation croisée à cinq folds.

Les indicateurs utilisés sont :
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Classification Report

**Résultats obtenus sur l'ensemble de test :**

| Modèle | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 76.00 % | 75.73 % |
| SVM | 75.67 % | 75.43 % |
| Random Forest | 70.00 % | 69.33 % |

La Logistic Regression est retenue comme modèle final en raison de ses résultats globaux légèrement supérieurs et de son interprétabilité.

Le pipeline complet, incluant le preprocessing et le modèle, est sauvegardé au format Pickle.

### 6. Automatisation avec Apache Airflow

L'orchestration du projet repose sur deux DAGs.

**DAG 1 : `mental_health_training`**

Il organise les opérations suivantes :
1. Préparation des données.
2. Entraînement des modèles.
3. Évaluation et comparaison des performances.
4. Sauvegarde du modèle final et des résultats.

**DAG 2 : `mental_health_prediction`**

Il permet de :
1. Charger les données à traiter.
2. Vérifier les données et le modèle sauvegardé.
3. Générer les prédictions.
4. Enregistrer les résultats dans un fichier CSV.

L'environnement Airflow est exécuté avec Docker Compose et utilise PostgreSQL pour la gestion de ses métadonnées.

### 7. Application interactive avec Streamlit

L'application MindCare AI permet de consulter les résultats du projet et de tester le modèle final.

Ses fonctionnalités principales sont :

**Consultation des performances**
- Affichage des métriques de classification.
- Comparaison des trois modèles.
- Affichage de la matrice de confusion.
- Analyse des résultats par classe.

**Test du modèle**
- Saisie des caractéristiques d'un étudiant.
- Génération d'une prédiction.
- Affichage de la catégorie estimée.

### 8. Architecture du dépôt

```text
Mental-Health-Risk-Prediction/
│
├── airflow/
│   ├── dags/
│   │   └── mental_health_training.py
│   ├── docker-compose.yaml
│   └── Dockerfile
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   ├── results/
│   │   └── evaluation_results.json
│   ├── student_mental_health.csv
│   ├── student_mental_health_clean.csv
│   └── mental_health_predictions.csv
│
├── models/
│   ├── logistic_regression_best.pkl
│   ├── svm_best.pkl
│   ├── random_forest_best.pkl
│   ├── preprocessing.joblib
│   └── mental_health_model.pkl
│
├── notebooks/
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training_evaluation.ipynb
│
├── src/
│   ├── data_preparation.py
│   ├── training.py
│   ├── evaluation.py
│   └── save_artifacts.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### 9. Installation et exécution

#### Prérequis

- Python
- Git
- Docker Desktop
- Docker Compose

#### Récupération du projet

```bash
git clone <URL_DU_DEPOT>
cd Mental-Health-Risk-Prediction
```

#### Création de l'environnement virtuel

```bash
python -m venv venv
```

Activation sous Windows :

```powershell
.\venv\Scripts\Activate.ps1
```

#### Installation des dépendances

```bash
python -m pip install -r requirements.txt
```

#### Lancement de l'application Streamlit

```bash
python -m pip install streamlit matplotlib
python -m streamlit run app/app.py
```

L'application est accessible sur :

`http://localhost:8501`

#### Lancement d'Airflow

```bash
docker compose -f airflow/docker-compose.yaml up -d --build
```

Interface Airflow :

`http://localhost:8081`

Pour arrêter les services :

```bash
docker compose -f airflow/docker-compose.yaml down
```

### 10. Limites et précautions d'utilisation

Les résultats obtenus dépendent du dataset utilisé, de ses caractéristiques et de sa représentativité.

Certaines classes, notamment `Medium`, présentent des difficultés de classification plus importantes.

Les corrélations observées ne démontrent pas de relations causales entre les variables.

**Ce projet est strictement pédagogique.** Les prédictions générées ne constituent ni un diagnostic médical ni un dépistage clinique. Elles ne doivent pas être utilisées pour prendre des décisions disciplinaires ou de sélection concernant les étudiants.