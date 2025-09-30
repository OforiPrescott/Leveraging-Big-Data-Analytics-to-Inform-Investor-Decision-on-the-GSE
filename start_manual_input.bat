@echo off
cd /d "%~dp0"
streamlit run manual_sentiment_interface.py --server.port 8502
pause
