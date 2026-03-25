import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier


class XGBLabelWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper para usar XGBoost com rótulos em texto e manter compatibilidade
    com o app, que espera previsões nas classes originais.
    """

    def __init__(
        self,
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        y_series = pd.Series(y)
        self.classes_ = np.array(sorted(y_series.unique()))
        self.class_to_int_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = y_series.map(self.class_to_int_)

        self.model_ = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=len(self.classes_),
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y_encoded)
        return self

    def predict(self, X):
        y_pred_encoded = self.model_.predict(X)
        return np.array([self.classes_[int(idx)] for idx in y_pred_encoded])

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
