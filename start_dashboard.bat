@echo off
echo ================================================
echo GSE SENTIMENT ANALYSIS DASHBOARD
echo ================================================
echo.
echo Starting dashboard...
echo.
echo Take screenshots of these 6 figures for your thesis:
echo • Figure 4.1: Data sources distribution (Executive Summary tab)
echo • Figure 4.2: Sentiment by source (Data Sources tab)
echo • Figure 4.3: Model performance (Model Performance tab)
echo • Figure 4.4: Correlation matrix (Correlation Studies tab)
echo • Figure 4.5: Confidence levels (Real-Time Predictions tab)
echo • Figure 4.6: Sector analysis (Time Series Analysis tab)
echo.
echo Press Ctrl+C to stop the dashboard
echo ================================================
echo.

python -m streamlit run working_dashboard.py

echo.
echo Dashboard stopped.
pause
