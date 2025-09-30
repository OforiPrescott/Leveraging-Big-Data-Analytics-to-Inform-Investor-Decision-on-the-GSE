#!/bin/bash
echo "Starting GSE Sentiment Analysis Dashboard..."
cd "$(dirname "$0")"
streamlit run gse_sentiment_analysis_system.py --server.port 8501
