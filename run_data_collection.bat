@echo off
cd /d "%~dp0"
python -c "from gse_sentiment_analysis_system import GSESentimentAnalyzer; analyzer = GSESentimentAnalyzer(); analyzer.run_daily_collection()"
pause
