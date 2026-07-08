import os, sys, warnings, itertools, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features", "feature_matrix.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TARGET_MAP = {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.05, 0.1],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0, 0.01, 0.1],
}

N_TRIALS = 50

def evaluate_params(params, X, y, cv):
    scores = []
    for train_idx, val_idx in cv.split(X):
        X_tr, X_v = X[train_idx], X[val_idx]
        y_tr, y_v = y[train_idx], y[val_idx]

        classes = np.array([0, 1, 2])
        cw = compute_class_weight("balanced", classes=classes, y=y_tr)
        sw = np.array([cw[i] for i in y_tr])

        model = xgb.XGBClassifier(
            **params,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, sample_weight=sw)
        y_pred = model.predict(X_v)
        scores.append(balanced_accuracy_score(y_v, y_pred))

    return np.mean(scores), np.std(scores)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
    feature_cols = [c for c in df.columns if c != "target"]
    X, y = df[feature_cols].values, df["target"].values

    print(f"Loaded {len(df)} rows, {len(feature_cols)} features")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Target: BEARISH={sum(y==0)}, FLAT={sum(y==1)}, BULLISH={sum(y==2)}")

    train_idx = df.index < "2024-01-01"
    test_idx = df.index >= "2024-01-01"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\nTrain: {len(X_train)} rows ({df.index[train_idx][0].date()} to {df.index[train_idx][-1].date()})")
    print(f"Test:  {len(X_test)} rows ({df.index[test_idx][0].date()} to {df.index[test_idx][-1].date()})\n")

    cv = TimeSeriesSplit(n_splits=3, test_size=120)

    param_names = list(PARAM_GRID.keys())
    results = []

    print(f"Random search: {N_TRIALS} trials, 5-fold time-series CV\n")
    start = time.time()

    for trial in range(N_TRIALS):
        params = {k: np.random.choice(v) for k, v in PARAM_GRID.items()}
        mean_score, std_score = evaluate_params(params, X_train, y_train, cv)
        results.append((mean_score, std_score, params))

        if (trial + 1) % 20 == 0:
            elapsed = time.time() - start
            best_so_far = max(results, key=lambda r: r[0])
            print(f"  Trial {trial+1}/{N_TRIALS} ({elapsed:.0f}s) | best so far: {best_so_far[0]:.4f} (+/-{best_so_far[1]:.4f})")

    elapsed = time.time() - start
    results.sort(key=lambda r: r[0], reverse=True)

    print(f"\n=== TUNING COMPLETE ({elapsed:.0f}s) ===\n")
    print(f"Top 5 parameter sets (CV balanced accuracy):\n")

    for rank, (mean, std, params) in enumerate(results[:5]):
        print(f"  #{rank+1}: {mean:.4f} (+/-{std:.4f})")
        for k, v in params.items():
            print(f"      {k:20s} = {v}")
        print()

    best_params = results[0][2]
    print(f"Best params selected:")
    for k, v in best_params.items():
        print(f"  {k:20s} = {v}")
    print()

    classes = np.array([0, 1, 2])
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    sw = np.array([cw[i] for i in y_train])

    final_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=3,
        random_state=42,
    )

    final_model.fit(X_train, y_train, sample_weight=sw)

    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n=== HELD-OUT TEST SET (2024-2026) ===")
    print(f"Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%)")

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("            BEARISH  FLAT    BULLISH")
    for i, (label, cls) in enumerate([("BEARISH", 0), ("FLAT", 1), ("BULLISH", 2)]):
        print(f"  {label:10s}  {cm[cls][0]:5d}   {cm[cls][1]:5d}   {cm[cls][2]:5d}")

    print("\n" + classification_report(y_test, y_pred, target_names=["BEARISH", "FLAT", "BULLISH"]))

    importances = final_model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    for _, row in feat_imp.iterrows():
        pct = int(row["importance"] * 100)
        print(f"  {row['feature']:20s} {row['importance']:.4f}  ({pct}%)")

    model_path = os.path.join(MODELS_DIR, "xgboost_model.json")
    final_model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=feat_imp.head(10), x="importance", y="feature", ax=axes[0])
    axes[0].set_title("Top 10 Feature Importances (Tuned)")

    top_cv_scores = [r[0] for r in results[:20]]
    axes[1].plot(range(1, 21), top_cv_scores, marker="o")
    axes[1].axhline(y=bal_acc, color="r", linestyle="--", label=f"Test: {bal_acc:.3f}")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Top 20 CV Scores vs Test Performance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, "feature_importance_tuned.png"), dpi=150)
    print(f"Plot saved to feature_importance_tuned.png")

if __name__ == "__main__":
    main()
