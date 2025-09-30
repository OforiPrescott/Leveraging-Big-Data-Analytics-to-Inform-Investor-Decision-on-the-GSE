@echo off
cd /d "%~dp0"
streamlit run gse_sentiment_analysis_system.py --server.port 8501
pause
