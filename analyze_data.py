import sqlite3
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

def analyze_database():
    """Analyze the GSE sentiment database structure and content"""
    print("=== GSE Sentiment Database Analysis ===\n")

    conn = sqlite3.connect('gse_sentiment.db')
    cursor = conn.cursor()

    # Get table information
    cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
    tables = cursor.fetchall()

    print(f"Database contains {len(tables)} tables:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f'PRAGMA table_info({table_name})')
        columns = cursor.fetchall()
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        count = cursor.fetchone()[0]

        print(f"\n{table_name.upper()} TABLE:")
        print(f"  Rows: {count}")
        print(f"  Columns: {[col[1] for col in columns]}")

        # Show sample data
        if count > 0:
            cursor.execute(f'SELECT * FROM {table_name} LIMIT 3')
            sample = cursor.fetchall()
            print(f"  Sample data: {sample[:2] if len(sample) > 0 else 'No data'}")

    conn.close()
    return tables

def load_sentiment_data():
    """Load sentiment data from database"""
    conn = sqlite3.connect('gse_sentiment.db')

    # Load sentiment data
    sentiment_df = pd.read_sql_query('''
        SELECT timestamp, source, sentiment_score, sentiment_label,
               company, confidence
        FROM sentiment_data
        ORDER BY timestamp DESC
    ''', conn)

    # Load manual sentiment data
    manual_df = pd.read_sql_query('''
        SELECT timestamp, company, sentiment_score, sentiment_label
        FROM manual_sentiment
        ORDER BY timestamp DESC
    ''', conn)

    conn.close()

    # Convert timestamps
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'], format='mixed', utc=True)
    manual_df['timestamp'] = pd.to_datetime(manual_df['timestamp'], format='mixed', utc=True)

    return sentiment_df, manual_df

def load_stock_data():
    """Load stock data from CSV files"""
    try:
        composite_df = pd.read_csv('GSE COMPOSITE INDEX.csv', header=None)
        financial_df = pd.read_csv('GSE FINANCIAL INDEX.csv', header=None)

        # Process composite data (similar to data loader)
        composite_data = []
        for idx, row in composite_df.iterrows():
            date_cell = str(row[23]) if len(row) > 23 else ""
            if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_cell.strip()):
                data_row = row[23:31].values
                composite_data.append(data_row)

        if composite_data:
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Turnover', 'Turnover_Value', 'Trades']
            composite_df = pd.DataFrame(composite_data, columns=columns)
            composite_df['Date'] = pd.to_datetime(composite_df['Date'], format='%d/%m/%Y', errors='coerce')

            # Clean numeric columns
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Turnover', 'Turnover_Value', 'Trades']
            for col in numeric_cols:
                composite_df[col] = composite_df[col].astype(str).str.replace('"', '').str.replace(',', '')
                composite_df[col] = pd.to_numeric(composite_df[col], errors='coerce')

            composite_df = composite_df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)

            # Add calculated columns
            composite_df['Price_Change'] = composite_df['Close'].diff()
            composite_df['Price_Change_Pct'] = composite_df['Close'].pct_change() * 100

            return composite_df

    except Exception as e:
        print(f"Error loading stock data: {e}")
        return None

def perform_eda(sentiment_df, manual_df, stock_df):
    """Perform comprehensive exploratory data analysis"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===\n")

    # 1. Sentiment Data Overview
    print("1. SENTIMENT DATA OVERVIEW")
    print("-" * 40)

    total_sentiment = len(sentiment_df)
    total_manual = len(manual_df)

    print(f"Total automated sentiment entries: {total_sentiment}")
    print(f"Total manual sentiment entries: {total_manual}")
    print(f"Total sentiment entries: {total_sentiment + total_manual}")

    if total_sentiment > 0:
        print("\nAutomated sentiment statistics:")
        print(f"  Date range: {sentiment_df['timestamp'].min()} to {sentiment_df['timestamp'].max()}")
        print(f"  Companies covered: {sentiment_df['company'].nunique()}")
        print(f"  Sources: {sentiment_df['source'].nunique()}")
        print(f"  Average sentiment score: {sentiment_df['sentiment_score'].mean():.3f}")
        print(f"  Sentiment score range: {sentiment_df['sentiment_score'].min():.3f} to {sentiment_df['sentiment_score'].max():.3f}")

        # Sentiment distribution
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()
        print("\nSentiment distribution:")
        for label, count in sentiment_counts.items():
            pct = (count / total_sentiment) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    # 2. Stock Data Overview
    print("\n2. STOCK DATA OVERVIEW")
    print("-" * 40)

    if stock_df is not None and not stock_df.empty:
        print(f"Stock data records: {len(stock_df)}")
        print(f"Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
        print(f"Price range: {stock_df['Close'].min():.2f} - {stock_df['Close'].max():.2f}")
        print(f"Average daily turnover: {stock_df['Turnover'].mean():.0f}")
        print(f"Average price change: {stock_df['Price_Change_Pct'].mean():.3f}%")

        # Trading days analysis
        stock_df['DayOfWeek'] = stock_df['Date'].dt.day_name()
        trading_days = stock_df['DayOfWeek'].value_counts()
        print("\nTrading activity by day:")
        for day, count in trading_days.items():
            print(f"  {day}: {count} days")

    # 3. Correlation Analysis
    print("\n3. CORRELATION ANALYSIS")
    print("-" * 40)

    if total_sentiment > 0:
        # Sentiment correlations
        numeric_cols = ['sentiment_score', 'confidence']
        sentiment_corr = sentiment_df[numeric_cols].corr()
        print("Sentiment feature correlations:")
        print(sentiment_corr)

        # Company sentiment analysis
        company_sentiment = sentiment_df.groupby('company').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'confidence': 'mean'
        }).round(3)

        print("\nCompany sentiment summary:")
        print(company_sentiment)

    # 4. Time Series Analysis
    print("\n4. TIME SERIES ANALYSIS")
    print("-" * 40)

    if total_sentiment > 0:
        # Daily sentiment aggregation
        daily_sentiment = sentiment_df.set_index('timestamp').resample('D')['sentiment_score'].agg(['mean', 'count'])
        daily_sentiment = daily_sentiment[daily_sentiment['count'] > 0]

        print(f"Days with sentiment data: {len(daily_sentiment)}")
        print(f"Average daily sentiment: {daily_sentiment['mean'].mean():.3f}")
        print(f"Daily sentiment volatility: {daily_sentiment['mean'].std():.3f}")

    return {
        'sentiment_stats': {
            'total_automated': total_sentiment,
            'total_manual': total_manual,
            'companies': sentiment_df['company'].nunique() if total_sentiment > 0 else 0,
            'sources': sentiment_df['source'].nunique() if total_sentiment > 0 else 0
        },
        'stock_stats': {
            'total_records': len(stock_df) if stock_df is not None else 0,
            'price_range': (stock_df['Close'].min(), stock_df['Close'].max()) if stock_df is not None and not stock_df.empty else (0, 0)
        }
    }

def create_visualizations(sentiment_df, manual_df, stock_df):
    """Create EDA visualizations"""
    print("\n=== GENERATING VISUALIZATIONS ===\n")

    # Create output directory
    import os
    if not os.path.exists('eda_plots'):
        os.makedirs('eda_plots')

    # 1. Sentiment Distribution
    if not sentiment_df.empty:
        plt.figure(figsize=(12, 8))

        # Sentiment score distribution
        plt.subplot(2, 3, 1)
        plt.hist(sentiment_df['sentiment_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Sentiment Score Distribution')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.axvline(sentiment_df['sentiment_score'].mean(), color='red', linestyle='--', label=f"Mean: {sentiment_df['sentiment_score'].mean():.3f}")
        plt.legend()

        # Sentiment label distribution
        plt.subplot(2, 3, 2)
        sentiment_labels = sentiment_df['sentiment_label'].value_counts()
        colors = ['red' if x == 'negative' else 'green' if x == 'positive' else 'gray' for x in sentiment_labels.index]
        sentiment_labels.plot(kind='bar', color=colors, alpha=0.7, edgecolor='black')
        plt.title('Sentiment Label Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        # Confidence distribution
        plt.subplot(2, 3, 3)
        plt.hist(sentiment_df['confidence'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')

        # Sentiment by source
        plt.subplot(2, 3, 4)
        source_sentiment = sentiment_df.groupby('source')['sentiment_score'].mean().sort_values()
        source_sentiment.plot(kind='barh', color='orange', alpha=0.7, edgecolor='black')
        plt.title('Average Sentiment by Source')
        plt.xlabel('Average Sentiment Score')

        # Sentiment by company
        plt.subplot(2, 3, 5)
        company_sentiment = sentiment_df.groupby('company')['sentiment_score'].mean().sort_values()
        company_sentiment.plot(kind='barh', color='purple', alpha=0.7, edgecolor='black')
        plt.title('Average Sentiment by Company')
        plt.xlabel('Average Sentiment Score')

        # Time series of sentiment
        plt.subplot(2, 3, 6)
        daily_sentiment = sentiment_df.set_index('timestamp').resample('D')['sentiment_score'].mean()
        daily_sentiment.plot(color='blue', linewidth=2)
        plt.title('Daily Average Sentiment Over Time')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('eda_plots/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved sentiment analysis plots to eda_plots/sentiment_analysis.png")

    # 2. Stock Data Visualizations
    if stock_df is not None and not stock_df.empty:
        plt.figure(figsize=(15, 10))

        # Price movement
        plt.subplot(2, 3, 1)
        plt.plot(stock_df['Date'], stock_df['Close'], linewidth=2, color='navy')
        plt.title('GSE Composite Index Price Movement')
        plt.xlabel('Date')
        plt.ylabel('Index Value')
        plt.xticks(rotation=45)

        # Daily returns distribution
        plt.subplot(2, 3, 2)
        plt.hist(stock_df['Price_Change_Pct'].dropna(), bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.title('Daily Return Distribution')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.axvline(stock_df['Price_Change_Pct'].mean(), color='blue', linestyle='--', label=f"Mean: {stock_df['Price_Change_Pct'].mean():.3f}%")
        plt.legend()

        # Trading volume
        plt.subplot(2, 3, 3)
        plt.bar(stock_df['Date'], stock_df['Turnover'], alpha=0.7, color='green', width=1)
        plt.title('Daily Trading Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.xticks(rotation=45)

        # Price change over time
        plt.subplot(2, 3, 4)
        plt.plot(stock_df['Date'], stock_df['Price_Change_Pct'], linewidth=1, color='orange')
        plt.title('Daily Price Changes')
        plt.xlabel('Date')
        plt.ylabel('Price Change (%)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(rotation=45)

        # Volume vs Price change
        plt.subplot(2, 3, 5)
        plt.scatter(stock_df['Turnover'], stock_df['Price_Change_Pct'], alpha=0.6, color='purple')
        plt.title('Volume vs Price Change')
        plt.xlabel('Trading Volume')
        plt.ylabel('Price Change (%)')

        # Rolling volatility
        plt.subplot(2, 3, 6)
        stock_df['Volatility'] = stock_df['Price_Change_Pct'].rolling(window=10).std()
        plt.plot(stock_df['Date'], stock_df['Volatility'], linewidth=2, color='brown')
        plt.title('10-Day Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('eda_plots/stock_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved stock analysis plots to eda_plots/stock_analysis.png")

    # 3. Correlation Heatmap
    if not sentiment_df.empty:
        # Create correlation matrix for numeric features
        numeric_features = ['sentiment_score', 'confidence']
        if len(sentiment_df) > 10:  # Need sufficient data
            corr_matrix = sentiment_df[numeric_features].corr()

            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Sentiment Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('eda_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("Saved correlation heatmap to eda_plots/correlation_heatmap.png")

def perform_feature_selection(sentiment_df, stock_df):
    """Perform feature selection for predictive modeling"""
    print("\n=== FEATURE SELECTION ANALYSIS ===\n")

    if sentiment_df.empty or stock_df is None or stock_df.empty:
        print("❌ Insufficient data for feature selection")
        return None

    # Prepare features for analysis
    print("Preparing features for selection...")

    # Create target variable (next day price movement)
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)
    stock_df['Target'] = (stock_df['Close'].shift(-1) > stock_df['Close']).astype(int)

    # Aggregate sentiment features by date
    sentiment_daily = sentiment_df.set_index('timestamp').resample('D').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'confidence': 'mean'
    }).fillna(0)

    # Flatten column names
    sentiment_daily.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 'confidence_mean']
    sentiment_daily = sentiment_daily.reset_index()

    # Merge with stock data
    # Convert timestamp to date for merging
    sentiment_daily_copy = sentiment_daily.copy()
    sentiment_daily_copy['timestamp'] = sentiment_daily_copy['timestamp'].dt.date
    sentiment_daily_copy['timestamp'] = pd.to_datetime(sentiment_daily_copy['timestamp'])

    merged_df = pd.merge(
        stock_df[['Date', 'Close', 'Price_Change_Pct', 'Turnover', 'Target']],
        sentiment_daily_copy,
        left_on='Date',
        right_on='timestamp',
        how='left'
    ).fillna(0)

    # Create additional features
    merged_df['price_ma_5'] = merged_df['Close'].rolling(5).mean()
    merged_df['price_ma_10'] = merged_df['Close'].rolling(10).mean()
    merged_df['rsi'] = 50  # Simplified RSI
    merged_df['volume_ratio'] = merged_df['Turnover'] / merged_df['Turnover'].rolling(10).mean()
    merged_df['price_change_1d'] = merged_df['Price_Change_Pct']
    merged_df['price_change_5d'] = merged_df['Close'].pct_change(5) * 100

    # Fill NaN values
    merged_df = merged_df.fillna(0)

    # Define feature columns
    feature_cols = [
        'sentiment_mean', 'sentiment_std', 'sentiment_count', 'confidence_mean',
        'price_ma_5', 'price_ma_10', 'rsi', 'volume_ratio',
        'price_change_1d', 'price_change_5d'
    ]

    X = merged_df[feature_cols]
    y = merged_df['Target']

    # Remove rows with NaN target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) < 50:
        print("❌ Insufficient data for feature selection")
        return None

    print(f"Feature selection dataset: {len(X)} samples, {len(feature_cols)} features")

    # 1. Correlation Analysis
    print("\n1. CORRELATION WITH TARGET")
    print("-" * 30)

    correlations = {}
    for col in feature_cols:
        corr = X[col].corr(y)
        correlations[col] = abs(corr)
        print(f"  {col}: {correlations[col]:.3f}")

    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print("\nTop correlated features:")
    for feature, corr in sorted_correlations[:5]:
        print(f"  {feature}: {corr:.3f}")

    # 2. Mutual Information
    print("\n2. MUTUAL INFORMATION SCORES")
    print("-" * 30)

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_dict = dict(zip(feature_cols, mi_scores))

    sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    print("Mutual information scores:")
    for feature, score in sorted_mi:
        print(f"  {feature}: {score:.4f}")

    # 3. Recursive Feature Elimination (RFE)
    print("\n3. RECURSIVE FEATURE ELIMINATION")
    print("-" * 30)

    rfe_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(rfe_model, n_features_to_select=5)
    rfe.fit(X, y)

    rfe_features = [feature for feature, selected in zip(feature_cols, rfe.support_) if selected]
    print(f"RFE selected features: {rfe_features}")

    # 4. Feature Importance from Random Forest
    print("\n4. RANDOM FOREST FEATURE IMPORTANCE")
    print("-" * 30)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    print("Feature importance scores:")
    for feature, importance in sorted_importance:
        print(f"  {feature}: {importance:.4f}")

    # Create feature selection summary
    feature_selection_results = {
        'correlation_analysis': sorted_correlations,
        'mutual_information': sorted_mi,
        'rfe_selected': rfe_features,
        'rf_importance': sorted_importance,
        'dataset_info': {
            'samples': len(X),
            'features': len(feature_cols),
            'target_distribution': y.value_counts().to_dict()
        }
    }

    # Save results
    import json
    with open('eda_plots/feature_selection_results.json', 'w') as f:
        json.dump(feature_selection_results, f, indent=2, default=str)

    print("\nFeature selection results saved to eda_plots/feature_selection_results.json")

    return feature_selection_results

def generate_summary_report(eda_stats, feature_results):
    """Generate comprehensive summary report"""
    print("\n=== EDA SUMMARY REPORT ===\n")

    report = {
        'data_overview': eda_stats,
        'key_findings': {},
        'recommendations': []
    }

    # Data Overview
    print("DATA OVERVIEW:")
    print("-" * 20)
    print(f"• Sentiment entries: {eda_stats['sentiment_stats']['total_automated'] + eda_stats['sentiment_stats']['total_manual']}")
    print(f"• Companies covered: {eda_stats['sentiment_stats']['companies']}")
    print(f"• News sources: {eda_stats['sentiment_stats']['sources']}")
    print(f"• Stock data records: {eda_stats['stock_stats']['total_records']}")

    # Key Findings
    print("\nKEY FINDINGS:")
    print("-" * 20)

    if feature_results:
        # Top correlated features
        top_corr = feature_results['correlation_analysis'][:3]
        print(f"• Top correlated features: {', '.join([f[0] for f in top_corr])}")

        # Most important features
        top_imp = feature_results['rf_importance'][:3]
        print(f"• Most important features: {', '.join([f[0] for f in top_imp])}")

        # RFE selected features
        rfe_features = feature_results['rfe_selected']
        print(f"• RFE selected features: {', '.join(rfe_features)}")

    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 20)

    recommendations = [
        "Focus on sentiment_mean and sentiment_count as primary predictors",
        "Include technical indicators (price_ma_5, volume_ratio) in models",
        "Consider ensemble methods combining correlation and importance-based features",
        "Monitor sentiment volatility as a risk indicator",
        "Expand data collection to increase sample size for better statistical power"
    ]

    for rec in recommendations:
        print(f"• {rec}")

    report['key_findings'] = {
        'top_correlated_features': [f[0] for f in feature_results['correlation_analysis'][:5]] if feature_results else [],
        'most_important_features': [f[0] for f in feature_results['rf_importance'][:5]] if feature_results else [],
        'rfe_selected_features': feature_results['rfe_selected'] if feature_results else []
    }
    report['recommendations'] = recommendations

    # Save report
    import json
    with open('eda_plots/eda_summary_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\nSummary report saved to eda_plots/eda_summary_report.json")

    return report

def main():
    """Main EDA pipeline"""
    print("Starting GSE Sentiment Analysis EDA Pipeline")
    print("=" * 50)

    # Analyze database structure
    tables = analyze_database()

    # Load data
    print("\nLoading data...")
    sentiment_df, manual_df = load_sentiment_data()
    stock_df = load_stock_data()

    print(f"Loaded {len(sentiment_df)} sentiment entries")
    print(f"Loaded {len(manual_df)} manual sentiment entries")
    print(f"Loaded {len(stock_df) if stock_df is not None else 0} stock records")

    # Perform EDA
    eda_stats = perform_eda(sentiment_df, manual_df, stock_df)

    # Create visualizations
    create_visualizations(sentiment_df, manual_df, stock_df)

    # Feature selection
    feature_results = perform_feature_selection(sentiment_df, stock_df)

    # Generate summary report
    summary_report = generate_summary_report(eda_stats, feature_results)

    print("\n" + "=" * 50)
    print("EDA Pipeline Complete!")
    print("Generated files:")
    print("• eda_plots/sentiment_analysis.png")
    print("• eda_plots/stock_analysis.png")
    print("• eda_plots/correlation_heatmap.png")
    print("• eda_plots/feature_selection_results.json")
    print("• eda_plots/eda_summary_report.json")

if __name__ == "__main__":
    main()