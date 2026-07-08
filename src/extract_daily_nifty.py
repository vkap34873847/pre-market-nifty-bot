import os, json, glob, pandas as pd
from datetime import datetime

SCALPING_DIR = r"C:\Users\vansh\Desktop\New folder (2)"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(SCALPING_DIR, "1min_*.json")))
print(f"Found {len(files)} 1-min files")

daily_data = []
for fpath in files:
    fname = os.path.basename(fpath)
    date_str = fname.replace("1min_", "").replace(".json", "")
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        continue
    try:
        with open(fpath, "r") as f:
            candles = json.load(f)
    except:
        continue
    if not candles:
        continue
    opens = [float(c["open"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]
    volumes = [float(c.get("volume", 0)) for c in candles]
    daily_data.append({
        "date": date_str,
        "Open": opens[0],
        "High": max(highs),
        "Low": min(lows),
        "Close": closes[-1],
        "Volume": sum(volumes),
    })

df = pd.DataFrame(daily_data)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")
df.to_csv(os.path.join(OUTPUT_DIR, "nifty_daily.csv"))
print(f"Saved {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
