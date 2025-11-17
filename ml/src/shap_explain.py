# ml/src/shap_explain.py
"""
Compute SHAP explanations for RandomForest pipeline.
Requires `shap` package.
"""
import joblib
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "../models/rf_pipeline.joblib"
DATA_PATH = "../datasets/loan_data.csv"

def main():
    pipe = joblib.load(os.path.abspath(MODEL_PATH))
    df = pd.read_csv(os.path.abspath(DATA_PATH))
    target = "loan_status"
    X = df.drop(columns=[target]) if target in df.columns else df

    # sample
    sample = X.sample(n=min(200, len(X)), random_state=42)

    # Extract preprocessed features if preprocessing step present
    # We will explain predictions on raw features via TreeExplainer using the underlying estimator & preprocessor.
    if hasattr(pipe, "named_steps") and "preproc" in pipe.named_steps and "clf" in pipe.named_steps:
        preproc = pipe.named_steps["preproc"]
        clf = pipe.named_steps["clf"]
        X_transformed = preproc.transform(sample)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        # summary plot (class 1)
        shap.summary_plot(shap_values[1], X_transformed, show=False)
        plt.tight_layout()
        out = os.path.abspath("../models/shap_summary.png")
        plt.savefig(out, bbox_inches='tight')
        print("Saved SHAP summary to:", out)
    else:
        print("Pipeline not in expected format (preproc + clf). Cannot compute SHAP.")

if __name__ == "__main__":
    main()
