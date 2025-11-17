# ml/src/create_feature_importance.py
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = "../models/rf_pipeline.joblib"
FEATURE_NAMES_PATH = "../models/feature_names.csv"

def main():
    model_path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    pipe = joblib.load(model_path)

    clf = None
    preproc = None
    if hasattr(pipe, "named_steps"):
        for name, step in pipe.named_steps.items():
            if hasattr(step, "feature_importances_"):
                clf = step
            if hasattr(step, "transformers") or name.lower() in ("preproc","preprocessor","preproc"):
                preproc = step

    if clf is None:
        for name, step in pipe.named_steps.items():
            if hasattr(step, "feature_importances_"):
                clf = step
                break
    if clf is None:
        raise RuntimeError("Could not find classifier with feature_importances_ in pipeline")

    importances = clf.feature_importances_

    feature_names = None
    if os.path.exists(os.path.abspath(FEATURE_NAMES_PATH)):
        feature_names = list(pd.read_csv(FEATURE_NAMES_PATH)["feature"])
    else:
        # try to infer using preprocessor if available
        try:
            from preprocess import get_feature_names_from_preprocessor
            feature_names = get_feature_names_from_preprocessor(preproc)
        except Exception:
            feature_names = None

    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"f_{i}" for i in range(len(importances))]

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)

    out_csv = os.path.abspath("../models/feature_importances_top.csv")
    fi_df.head(30).to_csv(out_csv, index=False)

    top_n = min(20, len(fi_df))
    plot_df = fi_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, top_n * 0.35)))
    plt.barh(range(len(plot_df)), plot_df["importance"].values)
    plt.yticks(range(len(plot_df)), plot_df["feature"].values)
    plt.xlabel("Feature importance (RandomForest)")
    plt.title("Top feature importances")
    plt.tight_layout()
    out_png = os.path.abspath("../models/feature_importance.png")
    plt.savefig(out_png)
    plt.show()
    print("Saved feature importance plot to:", out_png)
    print("Saved CSV to:", out_csv)

if __name__ == "__main__":
    main()
