import os, sys, pandas as pd, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

def load_csv(name):
    fp = os.path.join(RAW_DIR, f"{name}_daily.csv")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        lines = [f.readline().strip() for _ in range(3)]
    if "Ticker" in "".join(lines):
        df = pd.read_csv(fp, header=0, skiprows=[1, 2], index_col=0, parse_dates=True)
        df = df.dropna(how="all")
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            close_col = [c for c in df.columns if "close" in c.lower()][0]
        return df[[close_col]].rename(columns={close_col: f"close_{name}"})
    df = pd.read_csv(fp)
    if "Date" in df.columns:
        df = df.set_index(pd.to_datetime(df.pop("Date")))
    elif df.columns[0] == "" or "Unnamed" in str(df.columns[0]):
        df = df.set_index(pd.to_datetime(df.pop(df.columns[0])))
    close_col = "Close" if "Close" in df.columns else "close"
    if close_col not in df.columns:
        close_col = [c for c in df.columns if "close" in c.lower()][0]
    return df[[close_col]].rename(columns={close_col: f"close_{name}"})

def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    names = ["nifty", "vix", "usdinr", "crude", "nikkei", "hsi", "shanghai", "sp500", "nasdaq"]
    dfs = {}
    for name in names:
        d = load_csv(name)
        if d is None or d.empty:
            print(f"  SKIP [{name}] — not found or empty")
            continue
        dfs[name] = d
        print(f"  Loaded [{name}]: {len(d)} rows, {d.index[0].date()} to {d.index[-1].date()}")

    df = dfs["nifty"][["close_nifty"]].copy()

    nifty_raw = pd.read_csv(os.path.join(RAW_DIR, "nifty_daily.csv"), index_col=0, parse_dates=True)
    open_col = "Open" if "Open" in nifty_raw.columns else "open"
    df["open_nifty"] = nifty_raw[open_col].values

    for name in ["vix", "usdinr", "crude", "nikkei", "hsi", "shanghai", "sp500", "nasdaq"]:
        if name in dfs:
            df = df.join(dfs[name], how="left")

    # Pre-market features: known at 8:30 AM before Nifty opens at 9:15
    # gift_gap_pct: GIFT Nifty predicts open[T] / close[T-1] - 1 (~85% accurate)
    df["gift_gap_pct"] = df["open_nifty"] / df["close_nifty"].shift(1) - 1

    # Yesterday's Nifty return (close[T-1] / close[T-2] - 1)
    df["nifty_prev_ret"] = df["close_nifty"].pct_change().shift(1)

    # Yesterday's VIX level and change
    vix_s = df["close_vix"].shift(1)
    df["vix_level"] = vix_s
    df["vix_change"] = vix_s.pct_change()

    # Overnight changes (yesterday's close to day-before's close)
    df["usdinr_overnight"] = df["close_usdinr"].pct_change().shift(1)
    df["crude_overnight"] = df["close_crude"].pct_change().shift(1)

    # Asian market today returns (partially known at 8:30 AM)
    df["nikkei_today"] = df["close_nikkei"].pct_change()
    df["hsi_today"] = df["close_hsi"].pct_change()
    df["shanghai_today"] = df["close_shanghai"].pct_change()

    # Yesterday's US market returns (from US session that ended overnight)
    df["sp500_yest"] = df["close_sp500"].shift(1).pct_change()
    df["nasdaq_yest"] = df["close_nasdaq"].shift(1).pct_change()

    # 5-day rolling volatility (annualised)
    df["nifty_vol_5d"] = df["nifty_prev_ret"].rolling(5).std() * np.sqrt(252)

    # Day of week
    df["day_of_week"] = df.index.dayofweek

    # Target: today's close-to-close return (what we predict at 8:30 AM)
    target_return = df["close_nifty"].pct_change()
    df["target"] = 1
    df.loc[target_return > 0.003, "target"] = 2
    df.loc[target_return < -0.003, "target"] = 0

    feature_cols = [
        "gift_gap_pct", "vix_level", "vix_change", "usdinr_overnight",
        "crude_overnight", "nikkei_today", "hsi_today", "shanghai_today",
        "sp500_yest", "nasdaq_yest", "nifty_prev_ret", "nifty_vol_5d", "day_of_week",
    ]

    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}")

    feature_matrix = df[available + ["target"]].dropna()
    feature_matrix.to_csv(os.path.join(FEATURES_DIR, "feature_matrix.csv"))

    print(f"\nFeature matrix: {len(feature_matrix)} rows, {len(available)} features")
    print(f"Date range: {feature_matrix.index[0].date()} to {feature_matrix.index[-1].date()}")
    print(f"\nTarget distribution:")
    target_map = {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}
    for cls, label in target_map.items():
        count = (feature_matrix["target"] == cls).sum()
        print(f"  {label}: {count} ({count/len(feature_matrix)*100:.1f}%)")

if __name__ == "__main__":
    main()
