import os, sys, re, json, warnings, argparse
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gift_nifty_scraper import get_gift_nifty_gap



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")

TARGET_MAP = {0: "BEARISH", 1: "FLAT", 2: "BULLISH"}
TARGET_COLORS = {0: "\033[91m", 1: "\033[93m", 2: "\033[92m"}
RESET = "\033[0m"

FEATURE_COLS = [
    "gift_gap_pct", "vix_level", "vix_change", "usdinr_overnight",
    "crude_overnight", "nikkei_today", "hsi_today", "shanghai_today",
    "sp500_yest", "nasdaq_yest", "nifty_prev_ret", "nifty_vol_5d", "day_of_week",
]

YF_SYMBOLS = {
    "nifty": "^NSEI",
    "vix": "^INDIAVIX",
    "usdinr": "USDINR=X",
    "crude": "CL=F",
    "nikkei": "^N225",
    "hsi": "^HSI",
    "shanghai": "000001.SS",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
}

def fetch_yf(symbol, period="1mo"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty or len(df) < 1:
            return None
        df.index = pd.to_datetime(df.index.date)
        return df
    except Exception as e:
        print(f"fetch_yf({symbol}) failed: {e}")
        return None

def build_today_features(gift_gap_override=None):
    print("Fetching live data from yfinance...")
    dfs = {}
    for name, symbol in YF_SYMBOLS.items():
        d = fetch_yf(symbol)
        if d is not None:
            dfs[name] = d
            close_val = d["Close"].iloc[-1]
            print(f"  {name:10s} ({symbol:12s}): {d.index[-1].date()} close={close_val:.2f}" if not np.isnan(close_val) else f"  {name:10s} ({symbol:12s}): {d.index[-1].date()} close=N/A")
        else:
            print(f"  {name:10s} ({symbol:12s}): FAILED")

    today = date.today()
    print(f"\nToday: {today}")

    if "nifty" not in dfs:
        print("ERROR: Nifty data not available.")
        return None

    nifty_close = dfs["nifty"]["Close"]
    nifty_open = dfs["nifty"]["Open"]
    row = {}

    computed_gap = nifty_open.iloc[-1] / nifty_close.iloc[-2] - 1 if len(nifty_close) >= 2 else np.nan
    print("\nFetching GIFT Nifty gap from liveindex.org...")
    gift_pct, gift_data = get_gift_nifty_gap(use_cache=False)
    if gift_pct is not None and 0 < abs(gift_pct) < 3:
        print(f"  GIFT Nifty: {gift_data['price']} ({gift_pct:+.2f}%)")
        row["gift_gap_pct"] = gift_pct / 100
    else:
        reason = f"unrealistic ({gift_pct:+.2f}%)" if gift_pct is not None else f"failed ({gift_data})"
        print(f"  GIFT Nifty scraped {reason}. Using computed gap: {computed_gap*100:+.2f}%")
        row["gift_gap_pct"] = computed_gap

    if gift_gap_override is not None:
        row["gift_gap_pct"] = gift_gap_override / 100

    row["nifty_prev_ret"] = nifty_close.iloc[-2] / nifty_close.iloc[-3] - 1 if len(nifty_close) >= 3 else np.nan

    if "vix" in dfs:
        vix = dfs["vix"]["Close"]
        row["vix_level"] = vix.iloc[-2] if len(vix) >= 2 else np.nan
        row["vix_change"] = vix.iloc[-2] / vix.iloc[-3] - 1 if len(vix) >= 3 else np.nan
    else:
        row["vix_level"] = np.nan
        row["vix_change"] = np.nan

    for name, key in [("usdinr", "usdinr_overnight"), ("crude", "crude_overnight")]:
        if name in dfs:
            s = dfs[name]["Close"]
            row[key] = s.iloc[-2] / s.iloc[-3] - 1 if len(s) >= 3 else np.nan
        else:
            row[key] = np.nan

    for name, key in [("nikkei", "nikkei_today"), ("hsi", "hsi_today"), ("shanghai", "shanghai_today")]:
        if name in dfs:
            s = dfs[name]["Close"]
            idx = -1 if len(s) >= 2 and not pd.isna(s.iloc[-1]) else -2
            row[key] = s.iloc[idx] / s.iloc[idx-1] - 1 if len(s) >= 2 else np.nan
        else:
            row[key] = np.nan

    for name, key in [("sp500", "sp500_yest"), ("nasdaq", "nasdaq_yest")]:
        if name in dfs:
            s = dfs[name]["Close"]
            row[key] = s.iloc[-2] / s.iloc[-3] - 1 if len(s) >= 3 else np.nan
        else:
            row[key] = np.nan

    nifty_ret_series = nifty_close.pct_change().dropna()
    nifty_vol = nifty_ret_series.iloc[-min(5, len(nifty_ret_series)):].std() * np.sqrt(252) if len(nifty_ret_series) >= 3 else np.nan
    row["nifty_vol_5d"] = nifty_vol
    row["day_of_week"] = today.weekday()

    return row

def main():
    parser = argparse.ArgumentParser(description="Pre-market Nifty regime predictor")
    parser.add_argument("--gift-gap", type=float, default=None, help="GIFT Nifty gap in percent (e.g., 0.5 for +0.5%%)")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Run src/train_model.py first.")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    row = build_today_features(gift_gap_override=args.gift_gap)
    if row is None:
        sys.exit(1)

    feature_vector = np.array([[row.get(c, np.nan) for c in FEATURE_COLS]])
    feature_df = pd.DataFrame(feature_vector, columns=FEATURE_COLS)

    proba = model.predict_proba(feature_df)[0]
    pred_class = int(np.argmax(proba))
    pred_label = TARGET_MAP[pred_class]
    color = TARGET_COLORS[pred_class]

    gap_dir = "BULLISH" if row["gift_gap_pct"] > 0 else "BEARISH"
    layers_agree = (pred_label == "BULLISH" and row["gift_gap_pct"] > 0) or (pred_label == "BEARISH" and row["gift_gap_pct"] < 0)

    if args.json:
        output = {
            "date": str(date.today()),
            "features": {c: (row.get(c) if not (isinstance(row.get(c), float) and np.isnan(row.get(c))) else None) for c in FEATURE_COLS},
            "prediction": {
                "class": pred_label,
                "class_id": pred_class,
                "confidence": {TARGET_MAP[i]: round(float(proba[i]), 4) for i in range(3)},
                "max_confidence": float(max(proba)),
            },
            "gift_nifty": {
                "gap_pct": round(row["gift_gap_pct"] * 100, 2),
                "direction": gap_dir,
            },
            "layers_agree": layers_agree,
        }
        print(json.dumps(output, indent=2))
        return

    print("\n" + "=" * 60)
    print(f"  PRE-MARKET REGIME PREDICTOR - {date.today()}")
    print("=" * 60)
    print(f"\n  Feature values:")
    for c in FEATURE_COLS:
        val = row.get(c, np.nan)
        if c == "day_of_week":
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            print(f"    {c:20s} = {int(val) if not np.isnan(val) else 'N/A'} ({days[int(val)] if not np.isnan(val) else '?'})")
        elif isinstance(val, float) and np.isnan(val):
            print(f"    {c:20s} = N/A")
        elif abs(val) < 10:
            print(f"    {c:20s} = {val:+.4f}")
        else:
            print(f"    {c:20s} = {val:.2f}")

    print(f"\n  {'-' * 40}")
    print(f"  XGBoost Prediction: {color}{pred_label}{RESET}")
    print(f"  Confidence:")
    for i in range(3):
        cls = TARGET_MAP[i]
        bar_len = int(proba[i] * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"    {cls:8s} {proba[i]*100:5.1f}% {bar}")
    print(f"  {'-' * 40}")

    max_conf = max(proba)
    if pred_label == "BULLISH" and proba[2] >= 0.45:
        action = "BULLISH BIAS - Consider long intraday strategies"
    elif pred_label == "BEARISH" and proba[0] >= 0.45:
        action = "BEARISH BIAS - Consider short/hedge strategies"
    elif max_conf >= 0.45:
        action = f"{pred_label} bias - moderate confidence"
    else:
        action = "LOW CONFIDENCE - Regime not clear, use GIFT Nifty gap for opening direction"

    print(f"\n  Action: {action}")
    print(f"\n  GIFT Nifty Layer 1: opening direction signal at 8:30 AM")
    print(f"    gap = {row['gift_gap_pct']*100:+.2f}%")
    print(f"    direction: {gap_dir}")
    print()
    print(f"  Layer 1 (GIFT Nifty)  -> opens {gap_dir}")
    print(f"  Layer 2 (XGBoost EOD) -> {color}{pred_label}{RESET}")
    if layers_agree:
        print(f"  [OK] COMBINED SIGNAL: Both layers agree on {pred_label}")
    else:
        delta = abs(row["gift_gap_pct"] * 100)
        if delta > 0.5:
            print(f"  [!] SIGNAL CONFLICT: Layers disagree. GIFT Nifty gap ({row['gift_gap_pct']*100:+.1f}%) is decisive on open; human judgement needed for EOD.")
        else:
            print(f"  [-] NEUTRAL OPEN: Gap is small ({delta:.2f}%). XGBoost EOD signal takes priority.")

    print("=" * 60)

if __name__ == "__main__":
    main()
