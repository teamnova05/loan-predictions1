# ml/src/evaluate_via_api.py
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score

API_URL = "http://127.0.0.1:5000/predict"  # change if needed

def main(data_path="../datasets/loan_data.csv", target="loan_status"):
    df = pd.read_csv(data_path)
    y = df[target]
    X = df.drop(columns=[target])
    preds = []
    probs = []
    for _, row in X.iterrows():
        payload = row.to_dict()
        try:
            r = requests.post(API_URL, json=payload, timeout=10)
            r.raise_for_status()
            res = r.json()
            # adjust depending on API response format
            if "prediction" in res:
                preds.append(res["prediction"])
            elif "y_pred" in res:
                preds.append(res["y_pred"])
            elif "default" in res:
                preds.append(res["default"])
            else:
                # try numeric classes
                preds.append(res.get("class", None))
            if "probability" in res:
                probs.append(res["probability"])
            elif "proba" in res:
                probs.append(res["proba"])
            else:
                probs.append(None)
        except Exception as e:
            print("Request failed for row:", e)
            preds.append(None)
            probs.append(None)

    valid_idx = [i for i,v in enumerate(preds) if v is not None]
    y_valid = y.iloc[valid_idx].astype(int)
    preds_valid = [int(preds[i]) for i in valid_idx]

    print("Accuracy:", accuracy_score(y_valid, preds_valid))
    print("Classification report:")
    print(classification_report(y_valid, preds_valid))

if __name__ == "__main__":
    main()
