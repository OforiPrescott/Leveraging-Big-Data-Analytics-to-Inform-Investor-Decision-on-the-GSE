@echo off
cd /d "%~dp0"
python -m streamlit run working_dashboard.py --server.port 8501
pause
