import os, sys, json, asyncio, warnings, logging
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.predict_today import build_today_features, FEATURE_COLS, TARGET_MAP
from src.telegram_reporter import format_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "telegram_config.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "bot_listener.log")),
        logging.StreamHandler(),
    ],
)

def load_config():
    bot_token = os.environ.get("BOT_TOKEN")
    if not bot_token:
        if not os.path.exists(CONFIG_PATH):
            raise ValueError("Set BOT_TOKEN env var or create telegram_config.json")
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"bot_token": bot_token}

def generate_report():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    row = build_today_features()
    if row is None:
        return "Failed to build features."

    fv = np.array([[row.get(c, np.nan) for c in FEATURE_COLS]])
    proba = model.predict_proba(pd.DataFrame(fv, columns=FEATURE_COLS))[0]
    pc = int(np.argmax(proba))
    return format_report(row, proba, pc)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "GIFT Nifty Pre-Market Bot\n\n"
        "Commands:\n"
        "/report - Generate and send pre-market report\n"
        "/help - Show this message"
    )

async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("Generating report...")
    try:
        report_text = generate_report()
        await msg.edit_text(report_text, parse_mode="HTML")
    except Exception as e:
        await msg.edit_text(f"Error: {e}")
        logging.exception("Report generation failed")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/report - Generate and send pre-market report\n"
        "/help - Show this message"
    )

def main():
    try:
        config = load_config()
        token = config["bot_token"]
    except Exception as e:
        logging.error(f"Config error: {e}")
        sys.exit(1)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("help", help_cmd))

    logging.info("Bot started. Listening for commands...")
    app.run_polling()

if __name__ == "__main__":
    main()
