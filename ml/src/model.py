# ml/src/model.py
"""
Factory functions for classifiers used in training & production.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_random_forest(
    n_estimators=200,
    max_depth=12,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state
    )

def get_logistic_regression(
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
):
    return LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=random_state
    )
