# ml/src/train_default_pipeline.py
"""
Train baseline pipelines (Logistic Regression + Random Forest) on loan_data.csv
Updated to use robust preprocessor and save feature names for later use.
Run from ml/src or call with explicit --dataset/--save_dir paths.
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import build_preprocessor, get_feature_names_from_preprocessor

def load_data(path):
    df = pd.read_csv(path)
    return df

def get_feature_columns(df, target_col):
    cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return numeric_cols, categorical_cols

def save_confusion_matrix(y_true, y_pred, outpath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center',
                color='white' if cm.max()>0 and val>cm.max()/2 else 'black')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def main(dataset_path, save_dir, test_size=0.2, random_state=42):
    os.makedirs(save_dir, exist_ok=True)
    print("Loading dataset:", dataset_path)
    df = load_data(dataset_path)
    target = "loan_status"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset. Columns: {df.columns.tolist()}")

    print("Dataset shape:", df.shape)
    print("Target column:", target)
    print("Target value counts:")
    print(df[target].value_counts(dropna=False))

    y = df[target].copy()
    X = df.drop(columns=[target])

    if y.dtype == object or y.dtype.name == 'category':
        unique_vals = list(y.dropna().unique())
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            print("Mapping target using:", mapping)
            y = y.map(mapping)
        else:
            print("Warning: target has more than 2 unique string values. Proceeding without automatic binarization.")
    elif pd.api.types.is_numeric_dtype(y):
        unique_vals = np.unique(y.dropna())
        if set(unique_vals).issubset({0,1}):
            print("Numeric binary target detected (0/1).")
        else:
            print("Numeric target detected with values:", unique_vals[:10], "... (not strictly 0/1). Proceeding as-is.")

    stratify = y if len(np.unique(y.dropna())) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=stratify
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    numeric_cols, categorical_cols = get_feature_columns(df, target)
    print("Numeric cols:", numeric_cols)
    print("Categorical cols (sample up to 20):", categorical_cols[:20])

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    try:
        feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)
        if feature_names:
            fn_path = os.path.join(save_dir, "feature_names.csv")
            pd.DataFrame({"feature": feature_names}).to_csv(fn_path, index=False)
            print("Saved feature names to:", fn_path)
        else:
            print("Feature names could not be fully derived. Downstream plots may show generic names.")
    except Exception as e:
        print("Could not extract feature names:", e)

    models = {
        'logreg': LogisticRegression(max_iter=1000, random_state=int(random_state), class_weight='balanced', solver='lbfgs'),
        'rf': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=int(random_state), n_jobs=-1, class_weight='balanced')
    }

    results = {}
    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        pipe = Pipeline([('preproc', preprocessor), ('clf', clf)])
        try:
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            print(f"{name} CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            print(f"Cross-validation failed for {name}: {e}")
            cv_scores = np.array([np.nan])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {acc:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm_path = os.path.join(save_dir, f"{name}_confusion_matrix.png")
        try:
            save_confusion_matrix(y_test, y_pred, cm_path)
            print("Saved confusion matrix to", cm_path)
        except Exception as e:
            print("Could not save confusion matrix:", e)

        save_path = os.path.join(save_dir, f"{name}_pipeline.joblib")
        joblib.dump(pipe, save_path)
        print(f"Saved pipeline to {save_path}")

        results[name] = {'cv_scores': cv_scores, 'test_acc': acc}

    summary = []
    for name in results:
        cv_mean = float(np.nanmean(results[name]['cv_scores']))
        summary.append({'model': name, 'cv_mean_accuracy': cv_mean, 'test_accuracy': float(results[name]['test_acc'])})
    pd.DataFrame(summary).to_csv(os.path.join(save_dir, "training_summary.csv"), index=False)
    print("Training complete. Summary saved to", os.path.join(save_dir, "training_summary.csv"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to CSV dataset (e.g. ../datasets/loan_data.csv)")
    p.add_argument("--save_dir", default="../models", help="Directory to save models (relative to this script)")
    p.add_argument("--test_size", default=0.2, type=float)
    p.add_argument("--random_state", default=42, type=int)
    args = p.parse_args()
    main(args.dataset, args.save_dir, args.test_size, args.random_state)
