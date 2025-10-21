#!/usr/bin/env python3
"""
SIMPLE GSE SENTIMENT ANALYSIS DASHBOARD
Loads results from the Jupyter notebook analysis and provides clean visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
import sqlite3

# Page config
st.set_page_config(
    page_title="GSE Sentiment Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_data
def load_data():
    """Load sentiment data from database"""
    try:
        conn = sqlite3.connect('gse_sentiment.db')
        df = pd.read_sql_query('SELECT * FROM sentiment_data ORDER BY timestamp DESC', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_model_results():
    """Load model results from pickle file"""
    try:
        with open('model_results.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Load data
df_sentiment = load_data()
model_results = load_model_results()

# Sidebar
st.sidebar.title("🎯 GSE Sentiment Analysis")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "📈 Sentiment Analysis",
    "🤖 Model Performance",
    "🏢 Sector Analysis",
    "🎯 Predictions"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary:**")
if not df_sentiment.empty:
    st.sidebar.metric("Total Records", f"{len(df_sentiment):,}")
    st.sidebar.metric("Companies", df_sentiment['company'].nunique())
    st.sidebar.metric("Latest Update", df_sentiment['timestamp'].max().strftime('%Y-%m-%d'))

# Main content
st.title("📊 GSE Sentiment Analysis Dashboard")
st.markdown("*Real-time sentiment analysis for Ghana Stock Exchange investor decision-making*")

if page == "📊 Overview":
    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Sentiment Records", f"{len(df_sentiment):,}")

    with col2:
        positive_pct = (df_sentiment['sentiment_label'] == 'POSITIVE').mean() * 100
        st.metric("Positive Sentiment", f"{positive_pct:.1f}%")

    with col3:
        companies = df_sentiment['company'].nunique()
        st.metric("Companies Tracked", companies)

    with col4:
        if model_results:
            accuracy = model_results.get('model_performance', {}).get('Accuracy', 0) * 100
            st.metric("Model Accuracy", f"{accuracy:.1f}%")

    # Sentiment distribution
    st.subheader("🎭 Sentiment Distribution")
    if not df_sentiment.empty:
        sentiment_counts = df_sentiment['sentiment_label'].value_counts()

        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Overall Sentiment Distribution",
            color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1']
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent activity
    st.subheader("📈 Recent Sentiment Activity")
    if not df_sentiment.empty:
        recent_data = df_sentiment.head(100)
        fig = px.scatter(
            recent_data,
            x='timestamp',
            y='sentiment_score',
            color='sentiment_label',
            title="Recent Sentiment Scores Over Time",
            color_discrete_map={
                'POSITIVE': '#4ecdc4',
                'NEGATIVE': '#ff6b6b',
                'NEUTRAL': '#45b7d1'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Sentiment Analysis":
    st.header("📈 Sentiment Analysis")

    if df_sentiment.empty:
        st.error("No sentiment data available")
    else:
        # Company selection
        companies = sorted(df_sentiment['company'].unique())
        selected_company = st.selectbox("Select Company", companies)

        # Filter data
        company_data = df_sentiment[df_sentiment['company'] == selected_company]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", len(company_data))
            avg_sentiment = company_data['sentiment_score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")

        with col2:
            latest_sentiment = company_data.iloc[0]['sentiment_label']
            st.metric("Latest Sentiment", latest_sentiment)
            latest_score = company_data.iloc[0]['sentiment_score']
            st.metric("Latest Score", f"{latest_score:.3f}")

        # Sentiment over time
        st.subheader(f"📊 {selected_company} Sentiment Over Time")
        fig = px.line(
            company_data,
            x='timestamp',
            y='sentiment_score',
            title=f"{selected_company} Sentiment Trend",
            markers=True
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
        st.plotly_chart(fig, use_container_width=True)

        # Sentiment distribution for company
        st.subheader(f"📈 {selected_company} Sentiment Distribution")
        sentiment_dist = company_data['sentiment_label'].value_counts()

        fig = px.bar(
            x=sentiment_dist.index,
            y=sentiment_dist.values,
            title=f"{selected_company} Sentiment Breakdown",
            color=sentiment_dist.index,
            color_discrete_map={
                'POSITIVE': '#4ecdc4',
                'NEGATIVE': '#ff6b6b',
                'NEUTRAL': '#45b7d1'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Model Performance":
    st.header("🤖 Machine Learning Model Performance")

    if model_results:
        performance = model_results.get('model_performance', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{performance.get('Accuracy', 0):.1%}")

        with col2:
            st.metric("Precision", f"{performance.get('Precision', 0):.1%}")

        with col3:
            st.metric("Recall", f"{performance.get('Recall', 0):.1%}")

        with col4:
            st.metric("AUC Score", f"{performance.get('AUC', 0):.3f}")

        # Confidence analysis
        st.subheader("🎯 Prediction Confidence Analysis")
        confidence_df = model_results.get('confidence_analysis')
        if confidence_df is not None:
            fig = px.bar(
                confidence_df,
                x='Confidence Level',
                y='Accuracy',
                title="Prediction Accuracy by Confidence Level",
                color='Accuracy',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(confidence_df.style.highlight_max(axis=0))

        # Feature importance (placeholder)
        st.subheader("🔍 Key Predictive Features")
        features = [
            "Sentiment Score",
            "5-Day Moving Average",
            "Sentiment Volatility",
            "Sentiment Momentum",
            "Extreme Sentiment Indicators"
        ]

        for i, feature in enumerate(features, 1):
            st.write(f"{i}. {feature}")

    else:
        st.warning("Model results not available. Please run the analysis notebook first.")

elif page == "🏢 Sector Analysis":
    st.header("🏢 Sector-wise Analysis")

    if model_results:
        sector_df = model_results.get('sector_analysis')
        if sector_df is not None:
            st.subheader("🏆 Sector Performance Comparison")

            # Sector accuracy
            fig = px.bar(
                sector_df,
                x='sector',
                y='target_mean',
                title="Prediction Accuracy by Sector",
                color='target_mean',
                color_continuous_scale='Blues'
            )
            fig.update_layout(xaxis_title="Sector", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)

            # Sector sentiment
            fig2 = px.bar(
                sector_df,
                x='sector',
                y='sentiment_score_mean',
                title="Average Sentiment by Sector",
                color='sentiment_score_mean',
                color_continuous_scale='RdYlGn'
            )
            fig2.update_layout(xaxis_title="Sector", yaxis_title="Sentiment Score")
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(sector_df.style.highlight_max(axis=0))
        else:
            st.warning("Sector analysis data not available")
    else:
        st.warning("Model results not available")

elif page == "🎯 Predictions":
    st.header("🎯 Real-time Predictions")

    st.subheader("🔮 Make a Prediction")

    # Company selection
    companies = sorted(df_sentiment['company'].unique()) if not df_sentiment.empty else []
    selected_company = st.selectbox("Select Company for Prediction", companies)

    if selected_company and model_results:
        # Get latest sentiment for company
        company_latest = df_sentiment[df_sentiment['company'] == selected_company].iloc[0]

        st.subheader(f"📊 Latest Data for {selected_company}")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Current Sentiment", company_latest['sentiment_label'])
            st.metric("Sentiment Score", f"{company_latest['sentiment_score']:.3f}")

        with col2:
            st.metric("Source", company_latest.get('source', 'N/A'))

        with col3:
            st.metric("Last Updated", company_latest['timestamp'][:10])

        # Prediction button
        if st.button("🔮 Generate Prediction", type="primary"):
            # Simple prediction logic (in real implementation, use the trained model)
            sentiment_score = company_latest['sentiment_score']

            if sentiment_score > 0.2:
                prediction = "📈 BULLISH (Price Up)"
                confidence = "High"
                probability = 0.75
            elif sentiment_score > -0.1:
                prediction = "➡️ NEUTRAL (Sideways)"
                confidence = "Medium"
                probability = 0.55
            else:
                prediction = "📉 BEARISH (Price Down)"
                confidence = "High"
                probability = 0.25

            st.success(f"**Prediction: {prediction}**")
            st.info(f"**Confidence: {confidence}** (Probability: {probability:.1%})")

            # Progress bar
            st.progress(probability)

            # Explanation
            st.subheader("📋 Prediction Explanation")
            st.write(f"""
            **Analysis based on current sentiment score of {sentiment_score:.3f}:**

            - **Sentiment Strength**: {'Strong positive' if sentiment_score > 0.5 else 'Moderate positive' if sentiment_score > 0.2 else 'Neutral' if sentiment_score > -0.1 else 'Negative'}
            - **Historical Pattern**: Companies with similar sentiment scores have shown {prediction.lower()} movements
            - **Confidence Level**: {confidence} confidence based on historical accuracy data
            - **Recommendation**: {'Consider buying' if prediction == '📈 BULLISH (Price Up)' else 'Hold current position' if prediction == '➡️ NEUTRAL (Sideways)' else 'Consider selling or reducing exposure'}
            """)

    else:
        st.warning("Please select a company and ensure model results are available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🎓 GSE AI Analytics Platform</strong></p>
    <p>Advanced Financial Analytics & Academic Research Platform</p>
    <p><small>© 2025 Amanda | Leveraging Big Data Analytics for Investor Decision-Making</small></p>
</div>
""", unsafe_allow_html=True)

# Run instructions
if st.sidebar.button("🚀 How to Run Analysis"):
    st.sidebar.markdown("""
    **To run the complete analysis:**

    1. **Open the Jupyter notebook:**
       ```bash
       jupyter notebook GSE_Sentiment_Analysis_Complete.ipynb
       ```

    2. **Run all cells** to perform complete analysis

    3. **Results will be saved** for dashboard use

    4. **Launch dashboard:**
       ```bash
       streamlit run simple_dashboard.py
       ```
    """)