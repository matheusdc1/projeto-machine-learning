import json
import os

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from data_prep import (
    build_preprocessor,
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)


DATA_PATH = "data/dataset_ambiental.csv"
EXPERIMENT_MODEL_PATH = "models/best_model_smote.pkl"
EXPERIMENT_INFO_PATH = "models/model_info_smote.json"


def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo no conjunto de teste e retorna métricas principais.
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


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression_smote": LogisticRegression(max_iter=1000),
        "random_forest_smote": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "hist_gradient_boosting_smote": HistGradientBoostingClassifier(random_state=42),
    }

    best_model_name = None
    best_model_pipeline = None
    best_f1_macro = -1.0
    best_accuracy = -1.0

    print("\nResultados dos testes com SMOTE:\n")

    for model_name, model in models.items():
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)

        print(f"Modelo: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print("Classification Report:")
        print(metrics["report"])
        print("-" * 60)

        if metrics["f1_macro"] > best_f1_macro:
            best_f1_macro = metrics["f1_macro"]
            best_accuracy = metrics["accuracy"]
            best_model_name = model_name
            best_model_pipeline = pipeline

    joblib.dump(best_model_pipeline, EXPERIMENT_MODEL_PATH)

    model_info = {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "f1_macro": best_f1_macro,
    }

    with open(EXPERIMENT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print(f"Melhor modelo com SMOTE: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    print(f"F1 Macro: {best_f1_macro:.4f}")
    print(f"Modelo experimental salvo em: {EXPERIMENT_MODEL_PATH}")
    print(f"Informações experimentais salvas em: {EXPERIMENT_INFO_PATH}")


if __name__ == "__main__":
    main()
