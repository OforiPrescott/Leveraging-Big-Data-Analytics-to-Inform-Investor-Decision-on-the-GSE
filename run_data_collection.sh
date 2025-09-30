#!/bin/bash
echo "Running data collection..."
cd "$(dirname "$0")"
python -c "
from gse_sentiment_analysis_system import GSESentimentAnalyzer
analyzer = GSESentimentAnalyzer()
analyzer.run_daily_collection()
print('Data collection completed!')
"
