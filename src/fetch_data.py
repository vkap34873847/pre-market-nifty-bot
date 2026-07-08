import os, sys, yfinance as yf, pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

START = "2015-01-01"
END = datetime.now().strftime("%Y-%m-%d")

SYMBOLS = {
    "vix": "^INDIAVIX",
    "usdinr": "INR=X",
    "crude": "BZ=F",
    "nikkei": "^N225",
    "hsi": "^HSI",
    "shanghai": "000001.SS",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
}

LOCAL_NIFTY_FILE = os.path.join(RAW_DIR, "nifty_daily_from_scalping.csv")

def dl(name, symbol, force=False):
    fp = os.path.join(RAW_DIR, f"{name}_daily.csv")
    if os.path.exists(fp) and not force:
        print(f"  EXISTS [{name}] ({len(pd.read_csv(fp))} rows)")
        return
    print(f"  DL [{name}] ({symbol}) ...", end=" ", flush=True)
    try:
        df = yf.download(symbol, start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            print("EMPTY")
            return
        df.to_csv(fp)
        print(f"{len(df)} rows")
    except Exception as e:
        print(f"ERROR: {e}")

def build_nifty(force=False):
    out_fp = os.path.join(RAW_DIR, "nifty_daily.csv")
    if os.path.exists(out_fp) and not force:
        print(f"  EXISTS [nifty] ({len(pd.read_csv(out_fp))} rows)")
        return
    print("  BUILD [nifty] ...")
    yf_fp = os.path.join(RAW_DIR, "nifty_daily_yf.csv")
    if not os.path.exists(yf_fp) or force:
        print("    DL yfinance nifty ...", end=" ", flush=True)
        df_yf = yf.download("^NSEI", start=START, end=END, auto_adjust=True, progress=False)
        df_yf.to_csv(yf_fp)
        print(f"{len(df_yf)} rows")
    df_main = pd.read_csv(yf_fp, header=[0, 1], index_col=0, parse_dates=True)
    df_main.columns = df_main.columns.get_level_values(0)
    df_main = df_main[["Open", "High", "Low", "Close", "Volume"]]
    if os.path.exists(LOCAL_NIFTY_FILE):
        local = pd.read_csv(LOCAL_NIFTY_FILE, index_col=0, parse_dates=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            local[col] = local[col].astype(float)
        common = df_main.index.intersection(local.index)
        new_dates = local.index.difference(df_main.index)
        for dt in common:
            df_main.loc[dt] = local.loc[dt]
        if len(new_dates):
            df_main = pd.concat([df_main, local.loc[new_dates]])
        df_main = df_main.sort_index()
        print(f"    Merged: {len(df_main)} rows ({len(common)} overlapped, {len(new_dates)} new)")
    df_main.to_csv(out_fp)
    print(f"    Saved: {len(df_main)} rows ({df_main.index[0].date()} to {df_main.index[-1].date()})")

def main():
    force = "--force" in sys.argv
    build_nifty(force=force)
    for name, symbol in SYMBOLS.items():
        dl(name, symbol, force=force)

if __name__ == "__main__":
    main()
