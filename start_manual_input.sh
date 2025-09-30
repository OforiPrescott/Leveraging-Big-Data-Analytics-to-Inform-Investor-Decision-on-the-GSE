#!/bin/bash
echo "Starting Manual Sentiment Input Interface..."
cd "$(dirname "$0")"
streamlit run manual_sentiment_interface.py --server.port 8502
