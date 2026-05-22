@echo off
title Document AI Bot
cd /d "c:\Users\OEM\OneDrive\Desktop\Custom OCR and Document AI System for Handwritten Record Processing"

:restart
echo [%date% %time%] Starting bot...
venv\Scripts\python telegram_bot\bot.py
echo [%date% %time%] Bot stopped. Restarting in 5 seconds...
timeout /t 5 /nobreak
goto restart
