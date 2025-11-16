from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_random_forest(random_state=42):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

def get_logistic_regression(random_state=42):
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced',
        solver='lbfgs'
    )
