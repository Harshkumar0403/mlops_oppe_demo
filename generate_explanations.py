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


def generate_shap_explanations(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    artifacts_dir: str = ARTIFACTS_DIR,
):
    """
    Loads the trained model, calculates SHAP values for a SAMPLE of the test set,
    and saves global and individual explanation plots + a textual report.
    """
    print("--- Generating SHAP Explanations ---")

    # Ensure artifacts dir exists
    os.makedirs(artifacts_dir, exist_ok=True)

    # ---- Load model & data ----
    try:
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Prefer the model's view of features if available
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
            X = df[expected_features]
        else:
            # Fallback: drop target and non-feature columns
            X = df.drop(columns=["Class", "Time"], errors="ignore")

        y = df["Class"]
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return

    # ---- Train/val split (we just need test set) ----
    _, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Use a smaller sample of the test set for performance
    if len(X_test) > 2000:
        X_test_sample = X_test.sample(n=2000, random_state=42)
    else:
        X_test_sample = X_test

    print(f"Calculating SHAP values for a sample of {len(X_test_sample)} instances...")

    # ---- SHAP explainer ----
    # For tree-based models (DecisionTreeClassifier/XGBoost/etc.)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Handle binary classification: shap_values is usually a list [class0, class1]
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            shap_values_for_pos = shap_values[1]  # assume class '1' is positive
        else:
            shap_values_for_pos = shap_values[0]
    else:
        shap_values_for_pos = shap_values

    # ---- Global beeswarm summary plot ----
    print("Generating global SHAP summary (beeswarm) plot...")
    plt.figure()
    shap.summary_plot(
        shap_values_for_pos,
        X_test_sample,
        show=False,  # don't display, we are saving to file
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    summary_path = os.path.join(artifacts_dir, "shap_summary.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Global SHAP summary plot saved to: {summary_path}")

    # ---- Stacked force plot for all instances ----
    try:
        print("Generating stacked force plot (this may be large)...")
        # expected_value can be scalar or array depending on model
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(np.shape(expected_value)) > 0:
            # If array-like and multi-class, take the second component as positive
            if isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
                base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        p_all = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values_for_pos,
            features=X_test_sample,
            matplotlib=False,
        )
        force_path = os.path.join(artifacts_dir, "shap_force_plot_all.html")
        shap.save_html(force_path, p_all)
        print(f"Stacked force plot saved to: {force_path}")
    except Exception as e:
        print(f"⚠️ Could not generate stacked force plot: {e}")

    # ---- Textual SHAP report ----
    try:
        print("Generating textual SHAP report...")
        # Calculate feature importances from SHAP values
        importance_df = (
            pd.DataFrame(
                {
                    "feature": X_test_sample.columns,
                    "importance": np.abs(shap_values_for_pos).mean(axis=0),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        report_path = os.path.join(artifacts_dir, "shap_report.txt")
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

