import json
import os

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from data_prep import (
    build_preprocessor,
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)
from xgb_label_wrapper import XGBLabelWrapper


DATA_PATH = "data/dataset_ambiental.csv"
MODEL_OUTPUT_PATH = "models/best_model.pkl"
MODEL_INFO_PATH = "models/model_info.json"
EXPERIMENT_NAME = "qualidade_ambiental_classificacao"


def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo no conjunto de teste e retorna as métricas principais.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, zero_division=0)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "report": report,
    }


def log_model_params(model_name: str):
    if model_name in ["logistic_regression", "logistic_regression_balanced"]:
        mlflow.log_param("max_iter", 1000)
        if model_name == "logistic_regression_balanced":
            mlflow.log_param("class_weight", "balanced")

    if model_name == "random_forest":
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("random_state", 42)

    if model_name == "random_forest_balanced":
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("min_samples_leaf", 2)

    if model_name == "gradient_boosting":
        mlflow.log_param("random_state", 42)

    if model_name == "hist_gradient_boosting":
        mlflow.log_param("random_state", 42)

    if model_name == "xgboost":
        mlflow.log_param("n_estimators", 250)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.08)
        mlflow.log_param("subsample", 0.9)
        mlflow.log_param("colsample_bytree", 0.9)
        mlflow.log_param("random_state", 42)


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        "xgboost": XGBLabelWrapper(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=1,
        ),
    }

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_model_name = None
    best_model_pipeline = None
    best_f1_macro = -1.0
    best_accuracy = -1.0

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            mlflow.log_param("model_name", model_name)
            log_model_params(model_name)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_macro", metrics["f1_macro"])

            print(f"\nModelo: {model_name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Macro: {metrics['f1_macro']:.4f}")
            print("Classification Report:")
            print(metrics["report"])

            if metrics["f1_macro"] > best_f1_macro:
                best_f1_macro = metrics["f1_macro"]
                best_accuracy = metrics["accuracy"]
                best_model_name = model_name
                best_model_pipeline = pipeline

    joblib.dump(best_model_pipeline, MODEL_OUTPUT_PATH)

    model_info = {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "f1_macro": best_f1_macro,
    }

    with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Melhor modelo: {best_model_name}")
    print(f"Melhor Accuracy: {best_accuracy:.4f}")
    print(f"Melhor F1 Macro: {best_f1_macro:.4f}")
    print(f"Modelo salvo em: {MODEL_OUTPUT_PATH}")
    print(f"Informações do modelo salvas em: {MODEL_INFO_PATH}")


if __name__ == "__main__":
    main()
