import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_prep import (
    load_data,
    clean_data,
    split_features_target,
    split_train_test,
    build_preprocessor,
)


DATA_PATH = "data/dataset_ambiental.csv"
MODEL_OUTPUT_PATH = "models/best_model.pkl"
EXPERIMENT_NAME = "qualidade_ambiental_classificacao"


def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo no conjunto de teste e retorna as métricas principais.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "report": report
    }


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_model_name = None
    best_model_pipeline = None
    best_f1_macro = -1

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ]
            )

            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(pipeline, X_test, y_test)

            mlflow.log_param("model_name", model_name)

            if model_name == "logistic_regression":
                mlflow.log_param("max_iter", 1000)

            if model_name == "random_forest":
                mlflow.log_param("n_estimators", 200)
                mlflow.log_param("random_state", 42)

            if model_name == "gradient_boosting":
                mlflow.log_param("random_state", 42)

            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_macro", metrics["f1_macro"])

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"\nModelo: {model_name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Macro: {metrics['f1_macro']:.4f}")
            print("Classification Report:")
            print(metrics["report"])

            if metrics["f1_macro"] > best_f1_macro:
                best_f1_macro = metrics["f1_macro"]
                best_model_name = model_name
                best_model_pipeline = pipeline

    joblib.dump(best_model_pipeline, MODEL_OUTPUT_PATH)

    print("\n" + "=" * 50)
    print(f"Melhor modelo: {best_model_name}")
    print(f"Melhor F1 Macro: {best_f1_macro:.4f}")
    print(f"Modelo salvo em: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()