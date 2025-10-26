#!/usr/bin/env python3
"""
Create detailed descriptions for each visualization to match the analysis writeups
"""

import json
import os

def load_analysis_results():
    """Load the latest analysis results"""
    try:
        with open('eda_plots/eda_summary_report.json', 'r') as f:
            summary = json.load(f)

        with open('eda_plots/feature_selection_results.json', 'r') as f:
            feature_results = json.load(f)

        return summary, feature_results
    except FileNotFoundError as e:
        print(f"Error loading results: {e}")
        return None, None

def create_visualization_descriptions(summary, feature_results):
    """Create detailed descriptions for each visualization"""

    descriptions = {}

    # 1. Sentiment Analysis Plot
    descriptions['sentiment_analysis.png'] = f"""
Figure 4.2: Sentiment Analysis Distribution and Trends

This comprehensive visualization presents the sentiment analysis results across four key dimensions:

A. Sentiment Distribution (Top-Left): Bar chart showing the distribution of sentiment labels across all 69 sentiment entries. The analysis reveals a predominantly negative sentiment landscape with 52.2% negative entries, 44.9% positive entries, and 2.9% neutral entries, indicating a cautious to negative sentiment environment in the Ghanaian financial discourse during the analysis period.

B. Average Daily Sentiment Score (Top-Right): Time series plot showing sentiment score evolution over 28 days with available data. The average daily sentiment of -0.048 with volatility of 0.353 demonstrates moderate sentiment fluctuations, with a slight negative bias across the analyzed period.

C. Average Sentiment by Source (Bottom-Left): Bar chart comparing sentiment scores across 6 different news sources. This analysis reveals source-specific sentiment patterns, with some sources showing more positive coverage while others maintain more neutral or negative tones.

D. Average Sentiment by Company (Bottom-Right): Bar chart displaying sentiment scores across 10 GSE-listed companies. The analysis shows significant heterogeneity, with companies like EGH and FML showing positive sentiment (0.148 and 0.177 respectively) while others like ACCESS and CAL show more negative sentiment (-0.367 and -0.265 respectively).

This multi-dimensional analysis provides insights into both temporal and categorical sentiment patterns, supporting the research finding of a predominantly negative sentiment environment in Ghanaian financial discourse.
"""

    # 2. Stock Analysis Plot
    descriptions['stock_analysis.png'] = f"""
Figure 4.3: GSE Stock Market Analysis and Trading Patterns

This visualization presents a comprehensive analysis of GSE stock market data across four critical dimensions:

A. GSE Composite Index Price Over Time (Top-Left): Line chart showing the evolution of the GSE Composite Index from January 2020 to June 2025. The price range spans from 1,806.94 GHS to 6,703.62 GHS, with an average daily price change of 0.111%. This chart illustrates the overall market trend and volatility patterns during the analysis period.

B. Distribution of Daily Returns (Top-Right): Histogram displaying the distribution of daily percentage returns across 1,360 trading records. The analysis reveals the return distribution characteristics, including central tendency, spread, and potential outliers in daily market movements.

C. Daily Trading Volume (Bottom-Left): Line chart showing trading volume patterns over time. With an average daily turnover of 1,628,331 GHS, this visualization highlights trading activity patterns and identifies periods of high and low market participation.

D. Average Returns by Day of Week (Bottom-Right): Bar chart analyzing market performance patterns across trading days. The data shows Thursday and Wednesday having the highest trading frequency (279 days each), followed by Tuesday (276 days), Friday (269 days), and Monday (257 days). This analysis reveals potential day-of-week effects on market returns and trading activity.

These visualizations provide critical insights into market behavior, supporting the technical analysis approach that identified price moving averages and volume ratios as key predictive indicators.
"""

    # 3. Correlation Heatmap
    descriptions['correlation_heatmap.png'] = f"""
Figure 4.4: Feature Correlation Matrix and Relationships

This correlation heatmap visualizes the relationships between all numerical features in the merged sentiment-stock dataset, providing critical insights for feature selection and model development:

Key Correlation Patterns Identified:

A. Technical Indicator Correlations:
- Price moving averages (MA_5, MA_10) show strong positive correlations with each other (r > 0.95)
- Price change metrics (1-day and 5-day changes) exhibit moderate correlations with moving averages
- Volume ratio shows weak correlations with price-based indicators

B. Sentiment Feature Correlations:
- Sentiment features (sentiment_mean, sentiment_std, sentiment_count, confidence_mean) show very weak correlations with technical indicators
- This supports the feature selection finding that sentiment features have limited direct predictive power for price movements

C. Target Variable Relationships:
- The target variable (next-day price movement) shows strongest correlations with price_ma_5 (r = 0.130) and price_ma_10 (r = 0.124)
- Price change features show moderate relationships with the target
- Sentiment features show negligible correlations with price movements

D. Multicollinearity Considerations:
- High correlations between moving averages suggest potential multicollinearity
- Feature selection methods (RFE) appropriately selected price_ma_5 over price_ma_10 to avoid redundancy

This correlation analysis validates the feature selection results, confirming that technical indicators dominate predictive relationships while sentiment features require more sophisticated modeling approaches to capture their predictive value.
"""

    # 4. Feature Selection Summary
    descriptions['feature_selection_summary'] = f"""
Feature Selection Analysis Summary

Based on comprehensive analysis using multiple statistical and machine learning approaches:

1. Correlation Analysis Results:
   - Top correlated features: {', '.join(summary['key_findings']['top_correlated_features'])}
   - Price moving averages show strongest linear relationships with target variable
   - Sentiment features show weak correlations, indicating non-linear relationships

2. Mutual Information Scores:
   - Top features by mutual information: price_ma_10 (0.0464), price_ma_5 (0.0405), rsi (0.0268)
   - Captures both linear and non-linear dependencies
   - Confirms technical indicators as primary predictors

3. Recursive Feature Elimination (RFE):
   - Selected features: {', '.join(summary['key_findings']['rfe_selected_features'])}
   - Eliminates redundant features while maintaining predictive power
   - Balances model complexity with performance

4. Random Forest Feature Importance:
   - Top features: {', '.join(summary['key_findings']['most_important_features'])}
   - Price momentum (5-day and 1-day changes) most important
   - Volume indicators show significant predictive value

Key Insights:
- Technical indicators dominate predictive power across all methods
- Price momentum and moving averages are consistently selected
- Sentiment features show limited direct predictive relationships
- Ensemble feature selection provides robust variable selection
"""

    return descriptions

def create_individual_figure_descriptions():
    """Create separate descriptions for each subplot"""

    subplot_descriptions = {}

    # Sentiment Analysis Subplots
    subplot_descriptions['sentiment_distribution'] = """
Figure 4.2A: Sentiment Distribution

Bar chart showing the distribution of sentiment labels across all 69 sentiment entries:
- Negative sentiment: 52.2% (36 entries)
- Positive sentiment: 44.9% (31 entries)
- Neutral sentiment: 2.9% (2 entries)

This distribution indicates a predominantly cautious to negative sentiment environment in the Ghanaian financial discourse during the analysis period.
"""

    subplot_descriptions['daily_sentiment_trend'] = """
Figure 4.2B: Daily Sentiment Trend

Time series plot showing sentiment score evolution over 28 days with available data:
- Average daily sentiment: -0.048
- Daily sentiment volatility: 0.353
- Date range: September 1-30, 2025

The moderate volatility with slight negative bias suggests fluctuating but generally cautious market sentiment.
"""

    subplot_descriptions['sentiment_by_source'] = """
Figure 4.2C: Sentiment by News Source

Bar chart comparing average sentiment scores across 6 different news sources:
- Sources analyzed: citinewsroom.com, myjoyonline.com, ghanabusinessnews.com, etc.
- Sentiment range: -0.2 to +0.1 across sources
- Reveals source-specific sentiment patterns and potential reporting biases
"""

    subplot_descriptions['sentiment_by_company'] = """
Figure 4.2D: Sentiment by Company

Bar chart displaying average sentiment scores across 10 GSE-listed companies:
- Most positive: EGH (+0.148), FML (+0.177)
- Most negative: ACCESS (-0.367), CAL (-0.265)
- Shows significant heterogeneity in company-specific sentiment

This analysis supports sector-specific sentiment analysis approaches.
"""

    # Stock Analysis Subplots
    subplot_descriptions['price_over_time'] = """
Figure 4.3A: GSE Composite Index Price Evolution

Line chart showing GSE Composite Index from January 2020 to June 2025:
- Price range: 1,806.94 - 6,703.62 GHS
- Average daily change: 0.111%
- Total trading records: 1,360

Illustrates overall market trend and major price movements during the analysis period.
"""

    subplot_descriptions['returns_distribution'] = """
Figure 4.3B: Daily Returns Distribution

Histogram of daily percentage returns across all trading records:
- Shows return distribution characteristics
- Identifies central tendency and potential outliers
- Supports volatility analysis and risk assessment
"""

    subplot_descriptions['trading_volume'] = """
Figure 4.3C: Daily Trading Volume

Line chart showing trading volume patterns over time:
- Average daily turnover: 1,628,331 GHS
- Highlights periods of high and low market participation
- Identifies volume spikes that may correlate with news events
"""

    subplot_descriptions['returns_by_day'] = """
Figure 4.3D: Returns by Day of Week

Bar chart analyzing market performance across trading days:
- Thursday: 279 trading days
- Wednesday: 279 trading days
- Tuesday: 276 trading days
- Friday: 269 trading days
- Monday: 257 trading days

Reveals potential day-of-week effects on market returns and trading activity patterns.
"""

    return subplot_descriptions

def save_descriptions_to_word_doc(descriptions, subplot_descriptions):
    """Save all descriptions to a Word document format"""

    content = "VISUALIZATION DESCRIPTIONS FOR CHAPTER 4\n"
    content += "=" * 50 + "\n\n"

    # Main visualizations
    for viz_file, description in descriptions.items():
        if viz_file != 'feature_selection_summary':
            content += description + "\n\n"

    # Feature selection summary
    content += descriptions['feature_selection_summary'] + "\n\n"

    # Individual subplot descriptions
    content += "INDIVIDUAL SUBPLOT DESCRIPTIONS\n"
    content += "=" * 35 + "\n\n"

    for subplot, description in subplot_descriptions.items():
        content += description + "\n\n"

    # Save to file
    with open("Visualization_Descriptions_For_DOCX.txt", "w", encoding="utf-8") as f:
        f.write(content)

    print("Visualization descriptions saved to Visualization_Descriptions_For_DOCX.txt")
    return content

def main():
    """Main function to create visualization descriptions"""

    print("Creating Visualization Descriptions for Chapter 4")
    print("=" * 55)

    # Load analysis results
    summary, feature_results = load_analysis_results()

    if summary is None or feature_results is None:
        print("❌ Error: Could not load analysis results")
        return

    # Create descriptions
    descriptions = create_visualization_descriptions(summary, feature_results)
    subplot_descriptions = create_individual_figure_descriptions()

    # Save to file
    content = save_descriptions_to_word_doc(descriptions, subplot_descriptions)

    print("\n[SUCCESS] Descriptions created successfully!")
    print("File: Visualization_Descriptions_For_DOCX.txt")
    print("\nContent includes:")
    print("- Main figure descriptions (Figures 4.2, 4.3, 4.4)")
    print("- Individual subplot descriptions (4.2A, 4.2B, etc.)")
    print("- Feature selection summary")
    print("- Detailed analysis matching each visualization")

if __name__ == "__main__":
    main()