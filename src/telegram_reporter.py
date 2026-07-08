import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict_today import build_today_features, FEATURE_COLS, TARGET_MAP
from src.gift_nifty_scraper import get_gift_nifty_gap

import numpy as np
import xgboost as xgb
from datetime import date, datetime
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "telegram_config.json")

TARGET_COLORS_TG = {0: "\U0001F534", 1: "\U0001F7E1", 2: "\U0001F7E2"}

def load_config():
    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if bot_token and chat_id:
        return {"bot_token": bot_token, "chat_id": int(chat_id) if chat_id.isdigit() else chat_id}
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: Set BOT_TOKEN/CHAT_ID env vars or create {CONFIG_PATH}")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return json.load(f)

def format_report(row, proba, pred_class):
    pred_label = TARGET_MAP[pred_class]
    emoji = TARGET_COLORS_TG[pred_class]
    gap_dir = "BULLISH" if row["gift_gap_pct"] > 0 else "BEARISH"
    gap_arrow = "\U0001F7E2" if row["gift_gap_pct"] > 0 else "\U0001F534"
    layers_agree = (pred_label == "BULLISH" and row["gift_gap_pct"] > 0) or (pred_label == "BEARISH" and row["gift_gap_pct"] < 0)
    gap_pct = row["gift_gap_pct"] * 100

    today_str = date.today().strftime("%d %b %Y")
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][date.today().weekday()]

    lines = []
    lines.append(f"\U0001F4CA <b>Pre-Market Report  |  {today_str} ({weekday})</b>")
    lines.append("")
    lines.append(f"\U0001F3F0 <b>Layer 1 - GIFT Nifty Gap</b>")
    lines.append(f"  Gap: {gap_pct:+.2f}% {gap_arrow}")
    lines.append(f"  Direction: {gap_dir}")
    lines.append(f"  \U0001F535 Reliability: ~85%")
    lines.append("")
    lines.append(f"\U0001F9E0 <b>Layer 2 - XGBoost EOD Prediction</b>")
    lines.append(f"  Regime: {emoji} {pred_label}")
    lines.append(f"  Confidence: {max(proba)*100:.1f}%")
    lines.append(f"  BEARISH: {proba[0]*100:.0f}%  |  FLAT: {proba[1]*100:.0f}%  |  BULLISH: {proba[2]*100:.0f}%")
    lines.append("")

    if layers_agree:
        lines.append(f"\U00002705 <b>Combined Signal: {pred_label}</b>")
        lines.append("  Both layers agree on direction.")
    else:
        if gap_pct > 0.5 or gap_pct < -0.5:
            lines.append(f"\U000026A0 <b>Signal Conflict</b>")
            lines.append(f"  GIFT Nifty gap ({gap_pct:+.1f}%) is decisive on open.")
            lines.append("  Human judgement needed for EOD.")
        else:
            lines.append(f"\U0001F7E1 <b>Neutral Open</b>")
            lines.append("  Small gap. XGBoost EOD signal takes priority.")

    lines.append("")
    lines.append(f"\U0001F4CB <b>Key Features</b>")
    for c in FEATURE_COLS:
        val = row.get(c, np.nan)
        if c == "day_of_week":
            continue
        if isinstance(val, float) and np.isnan(val):
            continue
        if abs(val) < 10:
            lines.append(f"  <code>{c}</code>: {val:+.4f}")
        else:
            lines.append(f"  <code>{c}</code>: {val:.2f}")

    lines.append("")
    lines.append(f"\U0001F4E2 Auto-generated at {datetime.now().strftime('%H:%M')} IST")
    lines.append("#nifty #premarket #giftnifty")

    return "\n".join(lines)

def send_telegram(text, bot_token, chat_id):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    resp = requests.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        print(f"Telegram API error: {resp.status_code} {resp.text}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="GIFT Nifty Telegram Reporter")
    parser.add_argument("--dry-run", action="store_true", help="Print report to stdout instead of sending")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to config JSON")
    parser.add_argument("--gift-gap", type=float, default=None, help="Override GIFT Nifty gap %")
    args = parser.parse_args()

    config = load_config()
    bot_token = config.get("bot_token")
    chat_id = config.get("chat_id")
    if not bot_token or not chat_id:
        print("ERROR: bot_token and chat_id required in config")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found. Run src/train_model.py first.")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    row = build_today_features(gift_gap_override=args.gift_gap)
    if row is None:
        print("ERROR: Could not build features")
        sys.exit(1)

    feature_vector = np.array([[row.get(c, np.nan) for c in FEATURE_COLS]])
    feature_df = pd.DataFrame(feature_vector, columns=FEATURE_COLS)
    proba = model.predict_proba(feature_df)[0]
    pred_class = int(np.argmax(proba))

    report = format_report(row, proba, pred_class)

    if args.dry_run:
        report_path = os.path.join(PROJECT_ROOT, "telegram_report_dry.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {report_path}")
    else:
        ok = send_telegram(report, bot_token, chat_id)
        if ok:
            print(f"Report sent to Telegram chat {chat_id}")
        else:
            print("Failed to send report")

if __name__ == "__main__":
    main()
