import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Flush logs line by line
sys.stdout.reconfigure(line_buffering=True)


DATA_PATH = "data/v0/transactions_2022.csv"
MODEL_PATH = "artifacts/model.pkl"
ARTIFACTS_DIR = "artifacts"


def generate_shap_explanations():
    """
    Loads the trained model, calculates SHAP values for a SAMPLE of the test set,
    and saves:
      - Global SHAP bar summary plot
      - Textual feature-importance report
    (No heavy force plot to avoid OOM / Killed in CI.)
    """
    print("--- Generating SHAP Explanations ---")

    # -----------------------------
    # Load model and data
    # -----------------------------
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

        print(f"Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

        # Use the exact features the model was trained on if available
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            # Fallback: drop target and non-feature columns
            expected_features = [
                c for c in df.columns
                if c not in ("Class", "Time", "location")
            ]

        X = df[expected_features]
        y = df["Class"]
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return

    # -----------------------------
    # Train/Val split → use test
    # -----------------------------
    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use a smaller sample for performance
    if len(X_test) > 2000:
        X_test_sample = X_test.sample(n=2000, random_state=42)
    else:
        X_test_sample = X_test

    print(f"Calculating SHAP values for a sample of {len(X_test_sample)} instances...")

    # -----------------------------
    # SHAP computation
    # -----------------------------
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # For classification, shap_values is often a list [class0, class1]
    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            shap_values_pos = shap_values[1]  # assume class "1" = fraud
        else:
            shap_values_pos = shap_values[0]
    else:
        shap_values_pos = shap_values

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # -----------------------------
    # Global SHAP summary (bar)
    # -----------------------------
    print("Generating global summary (bar) plot...")
    plt.figure()
    shap.summary_plot(
        shap_values_pos,
        X_test_sample,
        plot_type="bar",
        show=False,
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    summary_path = os.path.join(ARTIFACTS_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Global summary plot saved to: {summary_path}")

    # -----------------------------
    # Textual SHAP report
    # -----------------------------
    try:
        print("Generating textual SHAP report...")
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_test_sample.columns,
                    "importance": np.abs(shap_values_pos).mean(axis=0),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        report_path = os.path.join(ARTIFACTS_DIR, "shap_report.txt")
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("🧠 MODEL EXPLAINABILITY RESULTS (SHAP)\n")
            f.write("=" * 60 + "\n")
            f.write("Top 10 Most Important Features:\n")
            f.write(importance_df.head(10).to_string(index=False))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("\n📋 Key Insights:\n")
            top_feature = importance_df.iloc[0]
            f.write(
                f"• Most predictive feature: {top_feature['feature']} "
                f"(Avg |SHAP|: {top_feature['importance']:.4f})\n"
            )
            f.write(
                "• Higher absolute SHAP values indicate stronger influence on fraud predictions.\n"
            )

        print(f"✅ Textual SHAP report saved to: {report_path}")
    except Exception as e:
        print(f"❌ Error during textual report generation: {e}")


if __name__ == "__main__":
    generate_shap_explanations()

