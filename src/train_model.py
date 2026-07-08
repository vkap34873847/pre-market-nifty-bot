import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "feature_matrix.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TARGET_MAP = {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")

    feature_cols = [c for c in df.columns if c != "target"]
    X, y = df[feature_cols].values, df["target"].values

    split_date = "2024-01-01"
    train_idx = df.index < split_date
    test_idx = df.index >= split_date

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {X_train.shape[0]} rows ({df.index[train_idx][0].date()} to {df.index[train_idx][-1].date()})")
    print(f"Test:  {X_test.shape[0]} rows ({split_date} to {df.index[test_idx][-1].date()})")

    classes = np.array([0, 1, 2])
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    sample_weights = np.array([class_weights[i] for i in y_train])

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.0,
        reg_alpha=0.1,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
        random_state=42,
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n=== TEST SET RESULTS ===")
    print(f"Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("            BEARISH  FLAT    BULLISH")
    for i, (label, cls) in enumerate([("BEARISH", 0), ("FLAT", 1), ("BULLISH", 2)]):
        print(f"  {label:10s}  {cm[cls][0]:5d}   {cm[cls][1]:5d}   {cm[cls][2]:5d}")

    print("\n" + classification_report(y_test, y_pred, target_names=["BEARISH", "FLAT", "BULLISH"]))

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    for _, row in feat_imp.iterrows():
        pct = int(row["importance"] * 100)
        print(f"  {row['feature']:20s} {row['importance']:.4f}  ({pct}%)")

    model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
    print(f"\nModel saved to {os.path.join(MODELS_DIR, 'xgboost_model.json')}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp.head(10), x="importance", y="feature")
    plt.title("Top 10 Feature Importances — XGBoost")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "feature_importance.png"), dpi=150)
    print(f"Feature importance plot saved to {os.path.join(PROJECT_ROOT, 'feature_importance.png')}")

if __name__ == "__main__":
    main()
