# ml/src/test_pipeline.py
import joblib
import pandas as pd

MODEL_PATH = "../models/rf_pipeline.joblib"
DATA_PATH = "../datasets/loan_data.csv"

def main():
    pipe = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["loan_status"]).iloc[:5]
    print("Sample input shape:", X.shape)
    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None
    print("Predictions:", preds)
    print("Probabilities:", probs)

if __name__ == "__main__":
    main()
