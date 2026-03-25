import json
import os
from pprint import pprint

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from data_prep import (
    build_preprocessor,
    clean_data,
    load_data,
    split_features_target,
    split_train_test,
)

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


DATA_PATH = "data/dataset_ambiental.csv"
EXPERIMENT_MODEL_PATH = "models/best_model_full_experiment.pkl"
EXPERIMENT_INFO_PATH = "models/model_info_full_experiment.json"


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


def build_search_estimator(X_train, y_train, with_smote=False):
    """
    Cria e ajusta uma busca aleatória para o HistGradientBoosting.
    """
    preprocessor = build_preprocessor(X_train)

    if with_smote:
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        )
        param_distributions = {
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_iter": [100, 150, 200, 300],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        )
        param_distributions = {
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_iter": [100, 150, 200, 300],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [10, 20, 30],
            "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


def main():
    os.makedirs("models", exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor(X_train)
    classes = sorted(y.unique())
    class_to_int = {label: idx for idx, label in enumerate(classes)}
    y_train_int = y_train.map(class_to_int)

    experiments = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest_balanced": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        ),
        "hist_gradient_boosting_smote": ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    if HAS_XGBOOST:
        experiments["xgboost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        num_class=len(classes),
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    rows = []
    best_model_name = None
    best_model = None
    best_f1 = -1.0
    best_acc = -1.0

    print("\nResultados dos modelos diretos:\n")

    for model_name, model in experiments.items():
        if model_name == "xgboost":
            model.fit(X_train, y_train_int)
            y_pred_int = model.predict(X_test)
            int_to_class = {idx: label for label, idx in class_to_int.items()}
            y_pred = pd.Series(y_pred_int).map(int_to_class)
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            report = classification_report(y_test, y_pred, zero_division=0)
            metrics = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "report": report,
            }
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

        rows.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
        )

        print(f"Modelo: {model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print("Classification Report:")
        print(metrics["report"])
        print("-" * 60)

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_acc = metrics["accuracy"]
            best_model_name = model_name
            best_model = model

    print("\nBuscando tuning para HistGradientBoosting sem SMOTE...\n")
    search_plain = build_search_estimator(X_train, y_train, with_smote=False)
    plain_metrics = evaluate_model(search_plain.best_estimator_, X_test, y_test)
    rows.append(
        {
            "model_name": "hist_gradient_boosting_tuned",
            "accuracy": plain_metrics["accuracy"],
            "f1_macro": plain_metrics["f1_macro"],
        }
    )
    print("Melhores parâmetros sem SMOTE:")
    pprint(search_plain.best_params_)
    print(f"Accuracy: {plain_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {plain_metrics['f1_macro']:.4f}")
    print("-" * 60)

    if plain_metrics["f1_macro"] > best_f1:
        best_f1 = plain_metrics["f1_macro"]
        best_acc = plain_metrics["accuracy"]
        best_model_name = "hist_gradient_boosting_tuned"
        best_model = search_plain.best_estimator_

    print("\nBuscando tuning para HistGradientBoosting com SMOTE...\n")
    search_smote = build_search_estimator(X_train, y_train, with_smote=True)
    smote_metrics = evaluate_model(search_smote.best_estimator_, X_test, y_test)
    rows.append(
        {
            "model_name": "hist_gradient_boosting_smote_tuned",
            "accuracy": smote_metrics["accuracy"],
            "f1_macro": smote_metrics["f1_macro"],
        }
    )
    print("Melhores parâmetros com SMOTE:")
    pprint(search_smote.best_params_)
    print(f"Accuracy: {smote_metrics['accuracy']:.4f}")
    print(f"F1 Macro: {smote_metrics['f1_macro']:.4f}")
    print("-" * 60)

    if smote_metrics["f1_macro"] > best_f1:
        best_f1 = smote_metrics["f1_macro"]
        best_acc = smote_metrics["accuracy"]
        best_model_name = "hist_gradient_boosting_smote_tuned"
        best_model = search_smote.best_estimator_

    results_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    print("\nResumo ordenado por F1 Macro:\n")
    print(results_df.to_string(index=False))

    joblib.dump(best_model, EXPERIMENT_MODEL_PATH)

    model_info = {
        "model_name": best_model_name,
        "accuracy": best_acc,
        "f1_macro": best_f1,
        "results": rows,
        "best_params_plain": search_plain.best_params_,
        "best_params_smote": search_smote.best_params_,
    }

    with open(EXPERIMENT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Melhor experimento geral: {best_model_name}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"F1 Macro: {best_f1:.4f}")
    print(f"Modelo salvo em: {EXPERIMENT_MODEL_PATH}")
    print(f"Informações salvas em: {EXPERIMENT_INFO_PATH}")


if __name__ == "__main__":
    main()
