import pandas as pd
import mlflow, mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Load dataset
df = pd.read_csv(
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Remove customerID
df.drop(
    "customerID",
    axis=1,
    inplace=True
)

# Handle TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = label_encoder.fit_transform(
            df[col]
        )

# Features and target
X = df.drop(
    "Churn",
    axis=1
)

y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# MLflow Experiment
mlflow.set_experiment(
    "Customer Churn Prediction"
)

for n_estimators in [50, 100, 200]:
    with mlflow.start_run():

    

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )
        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.sklearn.log_model(
            model,
            "random_forest_model"
        )

        print(
            f"n_estimators={n_estimators}, "
            f"accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, "
            f"recall={recall:.4f}, "
            f"f1={f1:.4f}"
        )