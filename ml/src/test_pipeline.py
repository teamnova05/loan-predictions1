# ml/src/test_pipeline.py
# (same as provided previously)
import joblib
import pandas as pd
import os

MODEL_PATH = "../models/rf_pipeline.joblib"
DATA_PATH = "../datasets/loan_data.csv"

def main():
    print("Loading model from:", os.path.abspath(MODEL_PATH))
    pipe = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    print("Loading dataset:", os.path.abspath(DATA_PATH))
    df = pd.read_csv(DATA_PATH)
    target_col = "loan_status"
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
    sample = X.head(5)
    print("Sample input (first 5 rows):")
    print(sample)
    preds = pipe.predict(sample)
    print("Predictions:", preds)
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(sample)
        print("Prediction probabilities:")
        print(probs)
    else:
        print("Model does not support predict_proba().")
    print("Test completed successfully.")

if __name__ == "__main__":
    main()
