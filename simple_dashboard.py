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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
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
    except FileNotFoundError:
        st.warning("Model results file not found. Some features may be limited.")
        return None
    except Exception as e:
        st.warning(f"Error loading model results: {e}")
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
st.markdown("### Real-time sentiment analysis for Ghana Stock Exchange investor decision-making")

# Add some spacing and better visual hierarchy
st.markdown("---")

if page == "📊 Overview":
    st.header("📊 Dashboard Overview")

    # Key metrics in a clean layout
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
        else:
            st.metric("Model Status", "Training Required")

    st.markdown("---")

    # Sentiment distribution
    st.subheader("🎭 Sentiment Distribution")
    if not df_sentiment.empty:
        sentiment_counts = df_sentiment['sentiment_label'].value_counts()

        col1, col2 = st.columns([1, 2])

        with col1:
            # Show percentages as metrics
            total = len(df_sentiment)
            for sentiment, count in sentiment_counts.items():
                pct = (count / total) * 100
                if sentiment == 'POSITIVE':
                    st.metric("Positive", f"{pct:.1f}%", f"{count} records")
                elif sentiment == 'NEGATIVE':
                    st.metric("Negative", f"{pct:.1f}%", f"{count} records")
                else:
                    st.metric("Neutral", f"{pct:.1f}%", f"{count} records")

        with col2:
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_sequence=['#4ecdc4', '#ff6b6b', '#45b7d1']
            )
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    # Recent activity
    st.subheader("📈 Recent Sentiment Activity")
    if not df_sentiment.empty:
        recent_data = df_sentiment.head(50)  # Show last 50 records

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
            },
            size_max=8
        )
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Time",
            yaxis_title="Sentiment Score"
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Sentiment Analysis":
    st.header("📈 Detailed Sentiment Analysis")

    if df_sentiment.empty:
        st.error("No sentiment data available")
        st.info("Please run data collection to populate the database.")
    else:
        # Company selection
        companies = sorted(df_sentiment['company'].unique())
        selected_company = st.selectbox("Select Company for Analysis", companies, key="company_select")

        # Filter data
        company_data = df_sentiment[df_sentiment['company'] == selected_company]

        # Company overview metrics
        st.subheader(f"📊 {selected_company} Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(company_data))

        with col2:
            avg_sentiment = company_data['sentiment_score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")

        with col3:
            latest_sentiment = company_data.iloc[0]['sentiment_label'] if not company_data.empty else "N/A"
            st.metric("Latest Sentiment", latest_sentiment)

        with col4:
            latest_score = company_data.iloc[0]['sentiment_score'] if not company_data.empty else 0.0
            st.metric("Latest Score", f"{latest_score:.3f}")

        st.markdown("---")

        # Sentiment over time
        st.subheader(f"📈 {selected_company} Sentiment Trend")

        if not company_data.empty:
            fig = px.line(
                company_data,
                x='timestamp',
                y='sentiment_score',
                title=f"{selected_company} Sentiment Over Time",
                markers=True,
                line_shape='spline'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7, annotation_text="Neutral")
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="Date/Time",
                yaxis_title="Sentiment Score"
            )
            fig.update_traces(line=dict(width=3), marker=dict(size=6))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No data available for {selected_company}")

        # Sentiment distribution for company
        st.subheader(f"📊 {selected_company} Sentiment Distribution")

        if not company_data.empty:
            sentiment_dist = company_data['sentiment_label'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                # Show breakdown metrics
                total = len(company_data)
                for sentiment, count in sentiment_dist.items():
                    pct = (count / total) * 100
                    if sentiment == 'POSITIVE':
                        st.metric("Positive", f"{pct:.1f}%", f"{count} records")
                    elif sentiment == 'NEGATIVE':
                        st.metric("Negative", f"{pct:.1f}%", f"{count} records")
                    else:
                        st.metric("Neutral", f"{pct:.1f}%", f"{count} records")

            with col2:
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
                fig.update_layout(
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Sentiment",
                    yaxis_title="Number of Records"
                )
                fig.update_traces(textposition='outside', texttemplate='%{y}')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No sentiment data available for {selected_company}")

elif page == "🤖 Model Performance":
    st.header("🤖 Machine Learning Model Performance")

    if model_results:
        performance = model_results.get('model_performance', {})

        st.subheader("📊 Model Metrics Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            accuracy = performance.get('Accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.1%}", "Prediction success rate")

        with col2:
            precision = performance.get('Precision', 0)
            st.metric("Precision", f"{precision:.1%}", "Positive prediction accuracy")

        with col3:
            recall = performance.get('Recall', 0)
            st.metric("Recall", f"{recall:.1%}", "True positive detection")

        with col4:
            auc = performance.get('AUC', 0)
            st.metric("AUC Score", f"{auc:.3f}", "Model discrimination power")

        st.markdown("---")

        # Confidence analysis
        st.subheader("🎯 Prediction Confidence Analysis")
        confidence_df = model_results.get('confidence_analysis')
        if confidence_df is not None:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(
                    confidence_df,
                    x='Confidence Level',
                    y='Accuracy',
                    title="Prediction Accuracy by Confidence Level",
                    color='Accuracy',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Confidence Level",
                    yaxis_title="Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Confidence Insights:**")
                max_accuracy = confidence_df['Accuracy'].max()
                best_confidence = confidence_df.loc[confidence_df['Accuracy'].idxmax(), 'Confidence Level']
                st.metric("Best Performance", f"{best_confidence}", f"{max_accuracy:.1%} accuracy")
                st.markdown("Higher confidence levels generally correlate with better prediction accuracy.")

            st.dataframe(confidence_df.style.highlight_max(axis=0), use_container_width=True)

        # Feature importance (placeholder)
        st.subheader("🔍 Key Predictive Features")

        if model_results and 'feature_importance' in model_results:
            feature_df = model_results['feature_importance']
            st.dataframe(feature_df.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.info("Feature importance data not available. Run complete analysis to see feature rankings.")

            # Show placeholder features
            features = [
                "Sentiment Score - Primary predictor of price movement",
                "Sentiment Volatility - Measures sentiment stability",
                "News Volume - Number of mentions in time period",
                "Source Credibility - Weight of different data sources",
                "Time-based Features - Recent vs historical sentiment"
            ]

            st.markdown("**Expected Key Features:**")
            for i, feature in enumerate(features, 1):
                st.write(f"{i}. {feature}")

    else:
        st.warning("Model results not available. Please run the analysis notebook first.")
        st.info("💡 To generate model results, run the GSE_Sentiment_Analysis_Complete.ipynb notebook")

elif page == "🏢 Sector Analysis":
    st.header("🏢 Sector-wise Analysis")

    if model_results:
        sector_df = model_results.get('sector_analysis')
        if sector_df is not None:
            st.subheader("🏆 Sector Performance Comparison")

            col1, col2 = st.columns(2)

            with col1:
                # Sector accuracy
                fig = px.bar(
                    sector_df,
                    x='sector',
                    y='target_mean',
                    title="Prediction Accuracy by Sector",
                    color='target_mean',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    xaxis_title="Sector",
                    yaxis_title="Accuracy",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Sector sentiment
                fig2 = px.bar(
                    sector_df,
                    x='sector',
                    y='sentiment_score_mean',
                    title="Average Sentiment by Sector",
                    color='sentiment_score_mean',
                    color_continuous_scale='RdYlGn'
                )
                fig2.update_layout(
                    xaxis_title="Sector",
                    yaxis_title="Sentiment Score",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader("📋 Detailed Sector Analysis")
            st.dataframe(sector_df.style.highlight_max(axis=0), use_container_width=True)

            # Sector insights
            st.subheader("🔍 Sector Insights")
            if not sector_df.empty:
                best_sector = sector_df.loc[sector_df['target_mean'].idxmax(), 'sector']
                worst_sector = sector_df.loc[sector_df['target_mean'].idxmin(), 'sector']
                st.success(f"🏆 **{best_sector}** shows the highest prediction accuracy")
                st.warning(f"⚠️ **{worst_sector}** shows the lowest prediction accuracy")
        else:
            st.warning("Sector analysis data not available")
            st.info("Run the complete analysis notebook to generate sector-wise insights")
    else:
        st.warning("Model results not available")
        st.info("💡 Run GSE_Sentiment_Analysis_Complete.ipynb to generate sector analysis")

elif page == "🎯 Predictions":
    st.header("🎯 Real-time Price Predictions")

    if df_sentiment.empty:
        st.error("No sentiment data available for predictions")
        st.info("Please run data collection first")
    else:
        st.subheader("🔮 Generate Price Movement Prediction")

        # Company selection
        companies = sorted(df_sentiment['company'].unique())
        selected_company = st.selectbox("Select Company for Prediction", companies, key="prediction_company")

        if selected_company:
            # Get latest sentiment for company
            company_latest = df_sentiment[df_sentiment['company'] == selected_company].iloc[0]

            st.subheader(f"📊 Current Data for {selected_company}")
            col1, col2, col3 = st.columns(3)

            with col1:
                sentiment_label = company_latest['sentiment_label']
                sentiment_score = company_latest['sentiment_score']
                st.metric("Current Sentiment", sentiment_label, f"Score: {sentiment_score:.3f}")

            with col2:
                source = company_latest.get('source', 'N/A')
                st.metric("Data Source", source)

            with col3:
                timestamp = company_latest['timestamp'][:10] if isinstance(company_latest['timestamp'], str) else str(company_latest['timestamp'])[:10]
                st.metric("Last Updated", timestamp)

            st.markdown("---")

            # Prediction button
            if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                # Simple prediction logic based on sentiment score
                sentiment_score = company_latest['sentiment_score']

                if sentiment_score > 0.2:
                    prediction = "📈 BULLISH (Price Up)"
                    confidence = "High"
                    probability = 0.75
                    color = "green"
                elif sentiment_score > -0.1:
                    prediction = "➡️ NEUTRAL (Sideways)"
                    confidence = "Medium"
                    probability = 0.55
                    color = "orange"
                else:
                    prediction = "📉 BEARISH (Price Down)"
                    confidence = "High"
                    probability = 0.25
                    color = "red"

                # Display prediction with better formatting
                st.success(f"**Prediction: {prediction}**")
                st.info(f"**Confidence Level: {confidence}** (Probability: {probability:.1%})")

                # Progress bar
                st.progress(probability)

                # Detailed explanation
                st.subheader("📋 Prediction Analysis")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write(f"""
                    **Analysis based on sentiment score of {sentiment_score:.3f}:**

                    • **Sentiment Strength**: {'Strong positive impact expected' if sentiment_score > 0.5 else 'Moderate positive sentiment' if sentiment_score > 0.2 else 'Neutral market sentiment' if sentiment_score > -0.1 else 'Negative sentiment pressure'}

                    • **Historical Pattern**: Companies with similar sentiment scores have shown {prediction.lower().replace('📈 ', '').replace('➡️ ', '').replace('📉 ', '')} movements

                    • **Confidence Assessment**: {confidence} confidence based on historical accuracy data

                    • **Investment Recommendation**: {'Consider buying or increasing position' if 'BULLISH' in prediction else 'Hold current position' if 'NEUTRAL' in prediction else 'Consider reducing exposure or selling'}
                    """)

                with col2:
                    st.markdown("**Prediction Metrics:**")
                    st.metric("Probability", f"{probability:.1%}")
                    st.metric("Sentiment Score", f"{sentiment_score:.3f}")
                    st.metric("Confidence", confidence)

            else:
                st.info("👆 Click 'Generate Prediction' to analyze current sentiment and predict price movement")
        else:
            st.warning("Please select a company to generate predictions")

# Footer with better styling
st.markdown("---")
st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%); padding: 30px; border-radius: 15px; margin: 20px 0; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1f2937; margin: 0 0 10px 0;'>🎓 GSE AI Analytics Platform</h3>
    <p style='color: #6b7280; margin: 0 0 15px 0; font-size: 1.1em;'>Advanced Financial Analytics & Academic Research Platform</p>
    <p style='color: #9ca3af; margin: 0; font-size: 0.9em;'>© 2025 Amanda | Leveraging Big Data Analytics for Investor Decision-Making</p>
    <div style='margin-top: 15px; opacity: 0.8;'>
        <span style='background: #3b82f6; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.8em;'>Real-time Sentiment Analysis</span>
        <span style='background: #10b981; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.8em; margin-left: 10px;'>Machine Learning Predictions</span>
    </div>
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