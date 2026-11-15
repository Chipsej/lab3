from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def _as_features(values: Any) -> np.ndarray:
    features = np.asarray(values, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError("Features must be a 1-D or 2-D array")
    return features


def _as_target(values: Any) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64).ravel()
    if target.size == 0:
        raise ValueError("Target must not be empty")
    return target


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0 
    train_norm = (train - mean) / scale
    test_norm = (test - mean) / scale
    return train_norm, test_norm, mean, scale


def _classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> ClassificationMetrics:
    y_true = np.asarray(y_true).ravel().astype(int)
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    if np.unique(y_true).size < 2:
        roc_auc = 0.5
    else:
        roc_auc = float(roc_auc_score(y_true, probabilities))

    return ClassificationMetrics(
        roc_auc=roc_auc,
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = _as_features(data.train_features)
    X_test = _as_features(data.test_features)
    y_train = _as_target(data.train_target)

    if X_train.shape[0] != y_train.size:
        raise ValueError("train_features and train_target must have the same number of rows")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("train and test features must have the same number of columns")
    if data.epochs < 0 or data.learning_rate <= 0:
        raise ValueError("epochs must be non-negative and learning_rate must be positive")

    X_train_n, X_test_n, feat_mean, feat_scale = _standardize(X_train, X_test)

    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    y_train_n = (y_train - y_mean) / y_scale

    n_samples, n_features = X_train_n.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    lr = float(data.learning_rate)
    loss_history: list[float] = []

    for _ in range(int(data.epochs)):
        y_pred_n = X_train_n @ weights + bias
        error = y_pred_n - y_train_n

        dw = (2.0 / n_samples) * (X_train_n.T @ error)
        db = (2.0 / n_samples) * float(error.sum())

        weights -= lr * dw
        bias -= lr * db

        y_pred = y_pred_n * y_scale + y_mean
        loss_history.append(float(np.mean((y_pred - y_train) ** 2)))

    original_weights = weights * y_scale / feat_scale
    original_bias = y_mean + bias * y_scale - float(feat_mean @ original_weights)

    test_predictions = X_test @ original_weights + original_bias

    return RegressionResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:

    X_train = _as_features(data.train_features)
    X_test = _as_features(data.test_features)
    y_train = _as_target(data.train_target)

    if X_train.shape[0] != y_train.size:
        raise ValueError("train_features and train_target must have the same number of rows")
    if X_test.shape[1] != X_train.shape[1]:
        raise ValueError("train and test features must have the same number of columns")
    if not np.isin(y_train, [0.0, 1.0]).all():
        raise ValueError("Logistic classifier supports binary targets encoded as 0 and 1")
    if data.epochs < 0 or data.learning_rate <= 0:
        raise ValueError("epochs must be non-negative and learning_rate must be positive")

    X_train_n, X_test_n, feat_mean, feat_scale = _standardize(X_train, X_test)

    n_samples, n_features = X_train_n.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    lr = float(data.learning_rate)
    loss_history: list[float] = []
    eps = 1e-15

    for _ in range(int(data.epochs)):
        logits = X_train_n @ weights + bias
        probs = _sigmoid(logits)

        loss = -np.mean(
            y_train * np.log(probs + eps) + (1.0 - y_train) * np.log(1.0 - probs + eps)
        )
        loss_history.append(float(loss))

        error = probs - y_train
        dw = (X_train_n.T @ error) / n_samples
        db = float(error.mean())

        weights -= lr * dw
        bias -= lr * db

    test_probs = _sigmoid(X_test_n @ weights + bias)

    original_weights = weights / feat_scale
    original_bias = bias - float(feat_mean @ original_weights)

    return ClassificationResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        probabilities=test_probs,
        predictions=(test_probs >= 0.5).astype(int),
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:

    X_train = _as_features(data.train_features)
    X_test = _as_features(data.test_features)
    y_train = _as_target(data.train_target).astype(int)
    y_test = _as_target(data.test_target).astype(int)

    if X_train.shape[0] != y_train.size:
        raise ValueError("train_features and train_target size mismatch")
    if X_test.shape[0] != y_test.size:
        raise ValueError("test_features and test_target size mismatch")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    tree = DecisionTreeClassifier(random_state=42)
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    )

    logistic.fit(X_train, y_train)
    tree.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    return BaselineComparisonResult(
        logistic_regression=_classification_metrics(
            y_test, logistic.predict_proba(X_test)[:, 1]
        ),
        decision_tree=_classification_metrics(
            y_test, tree.predict_proba(X_test)[:, 1]
        ),
        knn=_classification_metrics(
            y_test, knn.predict_proba(X_test)[:, 1]
        ),
    )
