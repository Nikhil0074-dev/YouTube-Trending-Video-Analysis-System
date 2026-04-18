"""ml/train_model.py
Train a Random Forest classifier and save the model + scaler.
Run: python ml/train_model.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score,
)

from config import MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH, RANDOM_STATE, TEST_SIZE
from backend.services.data_loader import load_processed
from backend.services.feature_engineering import build_feature_matrix

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def train():
    print("  Loading data …")
    df = load_processed()
    X  = build_feature_matrix(df)
    y  = df["trending"].values

    print(f"   Features: {X.shape[1]}  |  Samples: {len(y)}")
    print(f"   Class balance — trending: {y.sum()} / non-trending: {(1-y).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── Scale ─────────────────────────────────────────
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Models to compare ─────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=12,
                                                       random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                            random_state=RANDOM_STATE),
    }

    results = {}
    print("\n  Model Comparison:")
    print(f"{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("-" * 48)
    for name, model in models.items():
        use_scaled = name in ("Logistic Regression",)
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s  if use_scaled else X_test
        model.fit(Xtr, y_train)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {"model": model, "acc": acc, "auc": auc,
                         "scaled": use_scaled, "y_pred": y_pred}
        print(f"{name:<25} {acc:>10.4f} {auc:>10.4f}")

    # ── Pick best by AUC ─────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    best      = results[best_name]
    print(f"\n  Best model: {best_name}  (AUC = {best['auc']:.4f})")

    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, best["y_pred"],
                                 target_names=["Non-Trending", "Trending"]))

    # ── Feature importances (tree models) ────────────
    rf_model = results["Random Forest"]["model"]
    feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns)
    top10     = feat_imp.nlargest(10)
    print("\n  Top-10 Feature Importances (Random Forest):")
    for feat, imp in top10.items():
        print(f"   {feat:<35} {imp:.4f}")

    # ── Save ─────────────────────────────────────────
    save_model = results["Random Forest"]["model"]   # always save RF for explainability
    joblib.dump(save_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

    print(f"\n  Model saved  → {MODEL_PATH}")
    print(f"  Scaler saved → {SCALER_PATH}")

    # ── Save metrics JSON ─────────────────────────────
    import json
    metrics = {
        "best_model":   "Random Forest",
        "accuracy":     round(results["Random Forest"]["acc"], 4),
        "auc_roc":      round(results["Random Forest"]["auc"], 4),
        "top_features": {k: round(float(v), 4) for k, v in top10.items()},
        "all_models":   {k: {"accuracy": round(v["acc"], 4), "auc": round(v["auc"], 4)}
                         for k, v in results.items()},
    }
    metrics_path = os.path.join(os.path.dirname(MODEL_PATH), "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")
    return metrics


if __name__ == "__main__":
    train()
