# PowerShell script to run GSE dashboard without virtual environment issues
Write-Host "Starting GSE Sentiment Analysis Dashboard..." -ForegroundColor Green
python -m streamlit run working_dashboard.py --server.port 8501