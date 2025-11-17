# ml/src/evaluate_model.py
import argparse
import joblib
import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def find_file(patterns):
    import glob, os
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return sorted(matches, key=os.path.getmtime)[-1]
    return None

def ensure_dataframe(y):
    import pandas as pd, numpy as np
    if isinstance(y, (list, np.ndarray, pd.Series)):
        return pd.Series(y)
    return y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--save_report", default="./eval_report.txt")
    parser.add_argument("--save_predictions", default="./predictions.csv")
    args = parser.parse_args()

    COMMON_MODEL_PATHS = [
        "./ml/models/rf_pipeline.joblib",
        "./ml/models/*.joblib",
        "./models/*.joblib",
        "./models/*.pkl",
        "./ml/models/*.pkl"
    ]
    COMMON_DATA_PATHS = [
        "./ml/datasets/*.csv",
        "./ml/datasets/loan_data.csv",
        "./datasets/*.csv",
        "./data/*.csv",
        "./*.csv"
    ]

    model_path = args.model or find_file(COMMON_MODEL_PATHS)
    if model_path is None:
        print("No model file found. Provide --model")
        sys.exit(1)
    print("Using model:", model_path)
    model = joblib.load(model_path)

    data_path = args.data or find_file(COMMON_DATA_PATHS)
    if data_path is None:
        print("No data CSV found. Provide --data")
        sys.exit(1)
    print("Using data:", data_path)
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)

    target_col = args.target
    if not target_col:
        POSSIBLE_TARGET_NAMES = ["loan_status","Loan_Status","default","label","y","is_default"]
        for t in POSSIBLE_TARGET_NAMES:
            if t in df.columns:
                target_col = t
                break
    if not target_col:
        print("Could not detect target column. Columns:", df.columns.tolist())
        sys.exit(1)
    print("Target column:", target_col)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y))>1 else None)
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    try:
        preds = model.predict(X_test)
    except Exception as e:
        print("Model predict failed:", e)
        X_test_numeric = X_test.select_dtypes(include=[np.number])
        try:
            preds = model.predict(X_test_numeric)
            print("Predicted using numeric-only columns (fallback).")
        except Exception as e2:
            print("Fallback failed:", e2)
            sys.exit(1)

    preds = ensure_dataframe(preds)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
        except Exception:
            proba = None

    y_test_series = ensure_dataframe(y_test.reset_index(drop=True))

    acc = accuracy_score(y_test_series, preds)
    binary = len(np.unique(y_test_series))==2
    prec = precision_score(y_test_series, preds, average='binary' if binary else 'weighted', zero_division=0)
    rec = recall_score(y_test_series, preds, average='binary' if binary else 'weighted', zero_division=0)
    f1 = f1_score(y_test_series, preds, average='binary' if binary else 'weighted', zero_division=0)
    cm = confusion_matrix(y_test_series, preds)
    classif = classification_report(y_test_series, preds, zero_division=0)

    roc_auc = None
    if proba is not None and binary:
        try:
            roc_auc = roc_auc_score(y_test_series, proba[:,1])
        except Exception:
            roc_auc = None

    out_lines = []
    out_lines.append(f"Model file: {model_path}")
    out_lines.append(f"Dataset file: {data_path}")
    out_lines.append(f"Target column: {target_col}")
    out_lines.append("")
    out_lines.append("=== METRICS ===")
    out_lines.append(f"Accuracy: {acc:.4f}")
    out_lines.append(f"Precision: {prec:.4f}")
    out_lines.append(f"Recall: {rec:.4f}")
    out_lines.append(f"F1-score: {f1:.4f}")
    if roc_auc is not None:
        out_lines.append(f"ROC AUC: {roc_auc:.4f}")
    out_lines.append("")
    out_lines.append("Confusion Matrix:")
    out_lines.append(str(cm))
    out_lines.append("")
    out_lines.append("Classification Report:")
    out_lines.append(classif)

    report_text = "\n".join(out_lines)
    print(report_text)
    with open(args.save_report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Saved evaluation report to:", args.save_report)

    save_df = X_test.reset_index(drop=True).copy()
    save_df["y_true"] = y_test_series
    save_df["y_pred"] = preds
    if proba is not None:
        for i in range(proba.shape[1]):
            save_df[f"proba_class_{i}"] = proba[:,i]
    save_df.to_csv(args.save_predictions, index=False)
    print("Saved predictions to:", args.save_predictions)
