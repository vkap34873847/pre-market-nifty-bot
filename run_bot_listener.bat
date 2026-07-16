@echo off
set PROJECT_DIR=C:\Users\vansh\Desktop\Other Projects\pre_market_regime_classifier
cd /d "%PROJECT_DIR%"

if not exist "logs" mkdir logs

:restart
echo [%DATE% %TIME%] Starting bot listener... >> logs\bot_runner.log
python src\bot_listener.py >> logs\bot_runner.log 2>&1
echo [%DATE% %TIME%] Bot exited with code %ERRORLEVEL%. Restarting in 5s... >> logs\bot_runner.log
timeout /t 5 /nobreak >nul
goto restart
