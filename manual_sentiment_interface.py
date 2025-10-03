"""
Manual Sentiment Input Interface
Web interface for manual sentiment data entry for GSE companies
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ManualSentimentInterface:
    """Interface for manual sentiment data entry"""
    
    def __init__(self, db_path: str = "gse_sentiment.db"):
        self.db_path = db_path
        self.companies = self._load_gse_companies()
        self.news_types = [
            "earnings_report", "dividend_announcement", "management_change", 
            "regulatory_news", "partnership", "expansion", "acquisition",
            "market_rumor", "analyst_recommendation", "economic_policy",
            "sector_news", "competitor_news", "other"
        ]
        self.sentiment_levels = {
            "very_positive": {"score": 0.8, "label": "Very Positive 📈", "color": "#00ff00"},
            "positive": {"score": 0.4, "label": "Positive 📊", "color": "#90ee90"},
            "neutral": {"score": 0.0, "label": "Neutral ➖", "color": "#ffff00"},
            "negative": {"score": -0.4, "label": "Negative 📉", "color": "#ffa500"},
            "very_negative": {"score": -0.8, "label": "Very Negative 📉📉", "color": "#ff0000"}
        }
        
    def _load_gse_companies(self) -> List[Dict]:
        """Load GSE company information"""
        return [
            {'symbol': 'ACCESS', 'name': 'Access Bank Ghana Plc', 'sector': 'Banking'},
            {'symbol': 'CAL', 'name': 'CalBank PLC', 'sector': 'Banking'},
            {'symbol': 'CPC', 'name': 'Cocoa Processing Company', 'sector': 'Agriculture'},
            {'symbol': 'EGH', 'name': 'Ecobank Ghana PLC', 'sector': 'Banking'},
            {'symbol': 'EGL', 'name': 'Enterprise Group PLC', 'sector': 'Financial Services'},
            {'symbol': 'ETI', 'name': 'Ecobank Transnational Incorporation', 'sector': 'Banking'},
            {'symbol': 'FML', 'name': 'Fan Milk Limited', 'sector': 'Food & Beverages'},
            {'symbol': 'GCB', 'name': 'Ghana Commercial Bank Limited', 'sector': 'Banking'},
            {'symbol': 'GGBL', 'name': 'Guinness Ghana Breweries Plc', 'sector': 'Beverages'},
            {'symbol': 'GOIL', 'name': 'GOIL PLC', 'sector': 'Oil & Gas'},
            {'symbol': 'MTNGH', 'name': 'MTN Ghana', 'sector': 'Telecommunications'},
            {'symbol': 'RBGH', 'name': 'Republic Bank (Ghana) PLC', 'sector': 'Banking'},
            {'symbol': 'SCB', 'name': 'Standard Chartered Bank Ghana Ltd', 'sector': 'Banking'},
            {'symbol': 'SIC', 'name': 'SIC Insurance Company Limited', 'sector': 'Insurance'},
            {'symbol': 'SOGEGH', 'name': 'Societe Generale Ghana Limited', 'sector': 'Banking'},
            {'symbol': 'TOTAL', 'name': 'TotalEnergies Ghana PLC', 'sector': 'Oil & Gas'},
            {'symbol': 'UNIL', 'name': 'Unilever Ghana PLC', 'sector': 'Consumer Goods'},
            {'symbol': 'GLD', 'name': 'NewGold ETF', 'sector': 'Exchange Traded Fund'}
        ]
    
    def save_manual_sentiment(self, company: str, news_type: str, content: str, 
                             sentiment: str, impact_level: str, confidence: str,
                             user_id: str, source: str = "manual") -> bool:
        """Save manual sentiment entry to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure manual_sentiment table exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS manual_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    user_id TEXT,
                    company TEXT,
                    news_type TEXT,
                    content TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    impact_level TEXT,
                    confidence_level TEXT,
                    source TEXT DEFAULT 'manual'
                )
            ''')
            
            sentiment_score = self.sentiment_levels[sentiment]["score"]
            sentiment_label = sentiment
            
            cursor.execute('''
                INSERT INTO manual_sentiment 
                (timestamp, user_id, company, news_type, content, sentiment_score, 
                 sentiment_label, impact_level, confidence_level, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), user_id, company, news_type, content,
                sentiment_score, sentiment_label, impact_level, confidence, source
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Manual sentiment saved for {company}: {sentiment_label}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving manual sentiment: {str(e)}")
            return False
    
    def get_manual_sentiment_data(self, days_back: int = 30) -> pd.DataFrame:
        """Retrieve manual sentiment data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM manual_sentiment 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days_back)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving manual sentiment data: {str(e)}")
            return pd.DataFrame()
    
    def get_sentiment_summary(self, company: str = None, days_back: int = 30) -> Dict:
        """Get summary statistics of manual sentiment data"""
        try:
            df = self.get_manual_sentiment_data(days_back)
            
            if df.empty:
                return self._get_empty_summary()
            
            if company:
                df = df[df['company'] == company]
            
            if df.empty:
                return self._get_empty_summary()
            
            summary = {
                'total_entries': len(df),
                'avg_sentiment': df['sentiment_score'].mean(),
                'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
                'news_type_distribution': df['news_type'].value_counts().to_dict(),
                'company_distribution': df['company'].value_counts().to_dict(),
                'recent_trend': self._calculate_trend(df),
                'confidence_levels': df['confidence_level'].value_counts().to_dict() if 'confidence_level' in df.columns else {},
                'impact_levels': df['impact_level'].value_counts().to_dict() if 'impact_level' in df.columns else {}
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary: {str(e)}")
            return self._get_empty_summary()
    
    def _get_empty_summary(self) -> Dict:
        """Return empty summary when no data available"""
        return {
            'total_entries': 0,
            'avg_sentiment': 0.0,
            'sentiment_distribution': {},
            'news_type_distribution': {},
            'company_distribution': {},
            'recent_trend': 0.0,
            'confidence_levels': {},
            'impact_levels': {}
        }
    
    def _calculate_trend(self, df: pd.DataFrame) -> float:
        """Calculate sentiment trend over time"""
        if len(df) < 2:
            return 0.0
        
        # Sort by timestamp and calculate trend
        df_sorted = df.sort_values('timestamp')
        
        # Simple trend calculation using linear regression
        import numpy as np
        x = np.arange(len(df_sorted))
        y = df_sorted['sentiment_score'].values
        
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        return 0.0
    
    def render_input_form(self):
        """Render Streamlit form for manual sentiment input"""
        st.header("📝 Manual Sentiment Input")
        st.markdown("Add sentiment data based on news, rumors, or market observations about GSE companies.")
        
        with st.form("manual_sentiment_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Company selection
                company_options = {f"{comp['symbol']} - {comp['name']}": comp['symbol'] 
                                 for comp in self.companies}
                selected_company_display = st.selectbox(
                    "🏢 Select Company",
                    options=list(company_options.keys())
                )
                selected_company = company_options[selected_company_display]
                
                # News type
                news_type_labels = {
                    "earnings_report": "📊 Earnings Report",
                    "dividend_announcement": "💰 Dividend Announcement",
                    "management_change": "👥 Management Change",
                    "regulatory_news": "📋 Regulatory News",
                    "partnership": "🤝 Partnership/Deal",
                    "expansion": "🚀 Business Expansion",
                    "acquisition": "🏢 Acquisition/Merger",
                    "market_rumor": "💬 Market Rumor",
                    "analyst_recommendation": "📈 Analyst Recommendation",
                    "economic_policy": "🏛️ Economic Policy",
                    "sector_news": "🏭 Sector News",
                    "competitor_news": "🔄 Competitor News",
                    "other": "📰 Other News"
                }
                
                news_type_display = st.selectbox(
                    "📰 News/Information Type",
                    options=list(news_type_labels.keys()),
                    format_func=lambda x: news_type_labels[x]
                )
                
                # Sentiment selection
                sentiment_display = st.selectbox(
                    "💭 Sentiment",
                    options=list(self.sentiment_levels.keys()),
                    format_func=lambda x: self.sentiment_levels[x]["label"]
                )
                
                # Impact level
                impact_level = st.selectbox(
                    "⚡ Expected Impact Level",
                    options=["low", "medium", "high", "very_high"],
                    format_func=lambda x: {
                        "low": "🔹 Low Impact",
                        "medium": "🔸 Medium Impact", 
                        "high": "🔶 High Impact",
                        "very_high": "🔴 Very High Impact"
                    }[x]
                )
            
            with col2:
                # User information
                user_id = st.text_input(
                    "👤 Your Name/ID", 
                    value="",
                    placeholder="Enter your name or ID (optional)"
                )
                
                if not user_id:
                    user_id = "anonymous"
                
                # Confidence level
                confidence = st.selectbox(
                    "🎯 Confidence Level",
                    options=["low", "medium", "high"],
                    format_func=lambda x: {
                        "low": "🔹 Low Confidence",
                        "medium": "🔸 Medium Confidence",
                        "high": "🔶 High Confidence"
                    }[x],
                    index=1  # Default to medium
                )
                
                # Source of information
                source_type = st.selectbox(
                    "📡 Information Source",
                    options=["news_article", "social_media", "word_of_mouth", 
                           "official_announcement", "insider_info", "market_observation", "other"],
                    format_func=lambda x: {
                        "news_article": "📰 News Article",
                        "social_media": "📱 Social Media",
                        "word_of_mouth": "💬 Word of Mouth",
                        "official_announcement": "📢 Official Announcement",
                        "insider_info": "🔒 Insider Information",
                        "market_observation": "👁️ Market Observation",
                        "other": "❓ Other"
                    }[x]
                )
            
            # Content input
            st.markdown("### 📝 News Content/Description")
            content = st.text_area(
                "Describe the news, rumor, or information:",
                placeholder="Provide details about the news or information that affects this company...",
                height=120,
                help="Include as much detail as possible about the news or information that might affect the stock price."
            )
            
            # Additional context
            with st.expander("🔧 Additional Context (Optional)"):
                col3, col4 = st.columns(2)
                
                with col3:
                    expected_duration = st.selectbox(
                        "⏰ Expected Impact Duration",
                        options=["short_term", "medium_term", "long_term"],
                        format_func=lambda x: {
                            "short_term": "⚡ Short-term (days)",
                            "medium_term": "📊 Medium-term (weeks/months)",
                            "long_term": "📈 Long-term (months/years)"
                        }[x]
                    )
                
                with col4:
                    market_condition = st.selectbox(
                        "📊 Current Market Condition",
                        options=["bullish", "neutral", "bearish"],
                        format_func=lambda x: {
                            "bullish": "🟢 Bullish Market",
                            "neutral": "🟡 Neutral Market",
                            "bearish": "🔴 Bearish Market"
                        }[x],
                        index=1  # Default to neutral
                    )
            
            # Submit button
            submitted = st.form_submit_button(
                "💾 Submit Sentiment Data",
                type="primary"
            )
            
            if submitted:
                if content and len(content.strip()) > 10:
                    # Add additional context to content
                    enhanced_content = f"{content}\n\n[Additional Context: Impact Duration: {expected_duration}, Market Condition: {market_condition}, Source: {source_type}]"
                    
                    success = self.save_manual_sentiment(
                        company=selected_company,
                        news_type=news_type_display,
                        content=enhanced_content,
                        sentiment=sentiment_display,
                        impact_level=impact_level,
                        confidence=confidence,
                        user_id=user_id,
                        source="manual"
                    )
                    
                    if success:
                        st.success(f"✅ Successfully added {self.sentiment_levels[sentiment_display]['label']} sentiment for {selected_company}")
                        
                        # Show summary of what was added
                        with st.expander("📋 Submission Summary"):
                            st.write(f"**Company:** {selected_company_display}")
                            st.write(f"**Sentiment:** {self.sentiment_levels[sentiment_display]['label']}")
                            st.write(f"**News Type:** {news_type_labels[news_type_display]}")
                            st.write(f"**Impact Level:** {impact_level}")
                            st.write(f"**Confidence:** {confidence}")
                            st.write(f"**User:** {user_id}")
                    else:
                        st.error("❌ Error saving sentiment data. Please try again.")
                else:
                    st.error("⚠️ Please provide a meaningful description of the news or information.")
    
    def render_data_overview(self):
        """Render overview of submitted manual sentiment data"""
        st.header("📊 Manual Sentiment Data Overview")
        
        # Get summary data
        summary = self.get_sentiment_summary(days_back=30)
        
        if summary['total_entries'] == 0:
            st.info("📭 No manual sentiment data available yet. Start by adding some entries above!")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Entries", summary['total_entries'])
        
        with col2:
            avg_sentiment = summary['avg_sentiment']
            sentiment_emoji = "📈" if avg_sentiment > 0.1 else "📉" if avg_sentiment < -0.1 else "➖"
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}", delta=f"{sentiment_emoji}")
        
        with col3:
            trend = summary['recent_trend']
            trend_emoji = "📈" if trend > 0.01 else "📉" if trend < -0.01 else "➖"
            st.metric("Trend", f"{trend:.3f}", delta=f"{trend_emoji}")
        
        with col4:
            st.metric("Companies Covered", len(summary['company_distribution']))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            if summary['sentiment_distribution']:
                fig_sentiment = px.pie(
                    values=list(summary['sentiment_distribution'].values()),
                    names=list(summary['sentiment_distribution'].keys()),
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'very_positive': '#00ff00',
                        'positive': '#90ee90',
                        'neutral': '#ffff00',
                        'negative': '#ffa500',
                        'very_negative': '#ff0000'
                    }
                )
                st.plotly_chart(fig_sentiment, config={'responsive': True, 'displayModeBar': False})
        
        with col2:
            # Company distribution bar chart
            if summary['company_distribution']:
                fig_companies = px.bar(
                    x=list(summary['company_distribution'].keys()),
                    y=list(summary['company_distribution'].values()),
                    title="Entries by Company",
                    labels={'x': 'Company', 'y': 'Number of Entries'}
                )
                fig_companies.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_companies, config={'responsive': True, 'displayModeBar': False})
        
        # News types distribution
        if summary['news_type_distribution']:
            st.subheader("📰 News Types Distribution")
            fig_news = px.bar(
                x=list(summary['news_type_distribution'].values()),
                y=list(summary['news_type_distribution'].keys()),
                orientation='h',
                title="Distribution of News Types"
            )
            st.plotly_chart(fig_news, config={'responsive': True, 'displayModeBar': False})
        
        # Recent entries table
        st.subheader("📋 Recent Entries")
        recent_data = self.get_manual_sentiment_data(days_back=7)
        
        if not recent_data.empty:
            # Display formatted table
            display_df = recent_data[['timestamp', 'company', 'sentiment_label', 'news_type', 'user_id']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['Timestamp', 'Company', 'Sentiment', 'News Type', 'User']
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No recent entries to display.")
    
    def render_company_analysis(self):
        """Render company-specific sentiment analysis"""
        st.header("🏢 Company-Specific Analysis")
        
        # Company selection
        company_options = {comp['symbol']: f"{comp['symbol']} - {comp['name']}" 
                          for comp in self.companies}
        selected_company = st.selectbox(
            "Select Company for Analysis",
            options=list(company_options.keys()),
            format_func=lambda x: company_options[x]
        )
        
        # Get company-specific data
        summary = self.get_sentiment_summary(company=selected_company, days_back=90)
        
        if summary['total_entries'] == 0:
            st.info(f"📭 No manual sentiment data available for {selected_company} yet.")
            return
        
        # Company metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Entries", summary['total_entries'])
        
        with col2:
            avg_sentiment = summary['avg_sentiment']
            st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
        
        with col3:
            trend = summary['recent_trend']
            st.metric("Recent Trend", f"{trend:.3f}")
        
        # Time series of sentiment
        df = self.get_manual_sentiment_data(days_back=90)
        company_df = df[df['company'] == selected_company]
        
        if not company_df.empty:
            st.subheader("📈 Sentiment Over Time")
            
            fig_timeline = px.scatter(
                company_df,
                x='timestamp',
                y='sentiment_score',
                color='sentiment_label',
                size=[1] * len(company_df),  # Uniform size
                hover_data=['news_type', 'user_id'],
                title=f"Sentiment Timeline for {selected_company}",
                color_discrete_map={
                    'very_positive': '#00ff00',
                    'positive': '#90ee90',
                    'neutral': '#ffff00',
                    'negative': '#ffa500',
                    'very_negative': '#ff0000'
                }
            )
            
            # Add trend line
            if len(company_df) > 1:
                import numpy as np
                x_numeric = pd.to_numeric(company_df['timestamp'])
                z = np.polyfit(x_numeric, company_df['sentiment_score'], 1)
                p = np.poly1d(z)
                
                fig_timeline.add_scatter(
                    x=company_df['timestamp'],
                    y=p(x_numeric),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash')
                )
            
            st.plotly_chart(fig_timeline, config={'responsive': True, 'displayModeBar': False})
        
        # Recent entries for this company
        if not company_df.empty:
            st.subheader("📋 Recent Entries")
            recent_company = company_df.head(10)
            
            for _, entry in recent_company.iterrows():
                with st.expander(f"{entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {entry['sentiment_label']}"):
                    st.write(f"**News Type:** {entry['news_type']}")
                    st.write(f"**User:** {entry['user_id']}")
                    st.write(f"**Content:** {entry['content']}")

def run_manual_sentiment_app():
    """Run the manual sentiment input Streamlit app"""
    st.set_page_config(
        page_title="GSE Manual Sentiment Input",
        page_icon="📝",
        layout="wide"
    )
    
    interface = ManualSentimentInterface()
    
    st.title("📝 GSE Manual Sentiment Input System")
    st.markdown("*Capture market sentiment through manual input when automated scraping fails*")
    
    # Sidebar navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Input Form", "Data Overview", "Company Analysis"]
    )
    
    if page == "Input Form":
        interface.render_input_form()
    elif page == "Data Overview":
        interface.render_data_overview()
    elif page == "Company Analysis":
        interface.render_company_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built for GSE investors and analysts | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    run_manual_sentiment_app()