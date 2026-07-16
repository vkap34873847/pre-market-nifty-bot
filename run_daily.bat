@echo off
set PROJECT_DIR=C:\Users\vansh\Desktop\Other Projects\pre_market_regime_classifier
cd /d "%PROJECT_DIR%"

if not exist "logs" mkdir logs

set YEAR=%DATE:~6,4%
set MONTH=%DATE:~3,2%
set DAY=%DATE:~0,2%
set LOGFILE=logs\telegram_%YEAR%%MONTH%%DAY%.log

echo [%TIME%] Starting GIFT Nifty Telegram report >> "%LOGFILE%" 2>&1
python src\telegram_reporter.py >> "%LOGFILE%" 2>&1
set RESULT=%ERRORLEVEL%
echo [%TIME%] Finished (exit code %RESULT%) >> "%LOGFILE%" 2>&1
