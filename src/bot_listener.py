import os, sys, json, asyncio, warnings, logging, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone, timedelta

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from src.predict_today import build_today_features, FEATURE_COLS, TARGET_MAP
from src.telegram_reporter import format_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "telegram_config.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

SCHEDULE_HOUR = 8
SCHEDULE_MIN = 45

# Known NSE holidays for 2026 (non-exhaustive — expand as needed)
NSE_HOLIDAYS_2026 = {
    "2026-01-26",  # Republic Day
    "2026-03-27",  # Holi
    "2026-04-14",  # Dr Ambedkar Jayanti / Tamil New Year
    "2026-04-18",  # Good Friday
    "2026-05-01",  # Maharashtra Day
    "2026-08-17",  # Independence Day (observed)
    "2026-10-02",  # Gandhi Jayanti
    "2026-11-16",  # Diwali
    "2026-11-17",  # Diwali Balipratipada
    "2026-12-25",  # Christmas
}

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "bot_listener.log")),
        logging.StreamHandler(),
    ],
)

def load_config():
    bot_token = os.environ.get("BOT_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if not bot_token:
        if not os.path.exists(CONFIG_PATH):
            raise ValueError("Set BOT_TOKEN env var or create telegram_config.json")
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"bot_token": bot_token, "chat_id": int(chat_id) if chat_id else None}

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

async def send_scheduled_report(bot, chat_id):
    ist = timezone(timedelta(hours=5, minutes=30))
    while True:
        now_ist = datetime.now(ist)
        target = now_ist.replace(hour=SCHEDULE_HOUR, minute=SCHEDULE_MIN, second=0, microsecond=0)
        if now_ist >= target:
            target += timedelta(days=1)
        wait_sec = (target - now_ist).total_seconds()
        logging.info(f"Next scheduled report at {target.strftime('%H:%M')} IST (in {wait_sec/60:.0f} min)")
        await asyncio.sleep(wait_sec)

        today_str = datetime.now(ist).strftime("%Y-%m-%d")
        is_weekend = datetime.now(ist).weekday() >= 5
        is_holiday = today_str in NSE_HOLIDAYS_2026
        if is_weekend or is_holiday:
            reason = "weekend" if is_weekend else f"holiday ({today_str})"
            logging.info(f"Skipping report — {reason}")
            continue

        try:
            report_text = generate_report()
            await bot.send_message(chat_id=chat_id, text=report_text, parse_mode="HTML")
            logging.info("Scheduled report sent successfully.")
        except Exception as e:
            logging.exception(f"Scheduled report failed: {e}")

def scheduler_thread(bot, chat_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_scheduled_report(bot, chat_id))

def main():
    try:
        config = load_config()
        token = config["bot_token"]
        admin_chat_id = config.get("chat_id")
    except Exception as e:
        logging.error(f"Config error: {e}")
        sys.exit(1)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("help", help_cmd))

    if admin_chat_id:
        t = threading.Thread(target=scheduler_thread, args=(app.bot, admin_chat_id), daemon=True)
        t.start()
        logging.info(f"Scheduled daily report at {SCHEDULE_HOUR:02d}:{SCHEDULE_MIN:02d} IST")
    else:
        logging.warning("No CHAT_ID configured, skipping scheduled report.")

    def health_server():
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            def log_message(self, *a):
                pass
        port = int(os.environ.get("PORT", 10000))
        HTTPServer(("0.0.0.0", port), H).serve_forever()

    threading.Thread(target=health_server, daemon=True).start()
    logging.info("Health server started on port %s", os.environ.get("PORT", 10000))

    logging.info("Bot started. Listening for commands...")
    app.run_polling()

if __name__ == "__main__":
    main()
