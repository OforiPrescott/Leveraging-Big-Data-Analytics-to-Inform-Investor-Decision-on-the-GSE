#!/usr/bin/env python3
"""
Create proper visualizations using REAL data from the database
This addresses the user's concern about sample/generic data
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style for academic plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

class GSEVisualizationGenerator:
    """Generate proper visualizations using real GSE data"""

    def __init__(self):
        self.db_path = "gse_sentiment.db"
        self.sentiment_df = None
        self.stock_df = None
        self.output_dir = "proper_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_real_data(self):
        """Load real data from database and CSV files"""
        print("Loading real GSE data...")

        # Load sentiment data from database
        conn = sqlite3.connect(self.db_path)
        self.sentiment_df = pd.read_sql_query("""
            SELECT timestamp, source, sentiment_score, sentiment_label,
                   company, confidence, content
            FROM sentiment_data
        """, conn)
        conn.close()

        # Process sentiment data
        self.sentiment_df['timestamp'] = pd.to_datetime(self.sentiment_df['timestamp'], errors='coerce')
        self.sentiment_df['date'] = self.sentiment_df['timestamp'].dt.date
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])

        # Load and process stock data
        try:
            # Read the CSV and find actual data rows
            temp_df = pd.read_csv("GSE COMPOSITE INDEX.csv", header=None)

            # Find rows with actual trading data (containing dates)
            data_rows = []
            for idx, row in temp_df.iterrows():
                # Look for date patterns in the row
                for cell in row:
                    cell_str = str(cell).strip()
                    if '/' in cell_str and len(cell_str.split('/')) == 3:
                        try:
                            # Try to parse as date
                            pd.to_datetime(cell_str, format='%d/%m/%Y', errors='raise')
                            data_rows.append(row.values)
                            break
                        except:
                            continue

            if data_rows:
                self.stock_df = pd.DataFrame(data_rows)
                # Take only the relevant columns
                n_cols = min(8, len(self.stock_df.columns))
                self.stock_df = self.stock_df.iloc[:, :n_cols]
                self.stock_df.columns = ['Date', 'Open', 'High', 'Low', 'Close',
                                        'Turnover', 'Adj_Close', 'Trades'][:n_cols]

                # Convert data types
                self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'], format='%d/%m/%Y', errors='coerce')
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Turnover', 'Adj_Close', 'Trades']
                for col in numeric_cols[:len(self.stock_df.columns)-1]:  # Skip Date column
                    if col in self.stock_df.columns:
                        self.stock_df[col] = pd.to_numeric(self.stock_df[col], errors='coerce')

                # Remove invalid rows
                self.stock_df = self.stock_df.dropna(subset=['Date', 'Close'])

                print(f"Loaded {len(self.stock_df)} stock records")
            else:
                print("No valid stock data found")
                self.stock_df = pd.DataFrame()

        except Exception as e:
            print(f"Error loading stock data: {e}")
            self.stock_df = pd.DataFrame()

        print(f"Loaded {len(self.sentiment_df)} sentiment entries")
        print(f"Companies: {sorted(self.sentiment_df['company'].unique())}")
        print(f"Sources: {sorted(self.sentiment_df['source'].unique())}")

    def create_sentiment_distribution_viz(self):
        """Create sentiment distribution with real company names"""
        print("Creating sentiment distribution visualization...")

        plt.figure(figsize=(12, 8))

        # Get sentiment distribution
        sentiment_counts = self.sentiment_df['sentiment_label'].value_counts()

        # Create colorful bar chart
        colors = {'negative': '#e74c3c', 'positive': '#27ae60', 'neutral': '#f39c12'}
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values,
                      color=[colors.get(label, '#95a5a6') for label in sentiment_counts.index],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

        plt.title('Figure 4.2A: Sentiment Distribution Across GSE Companies',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sentiment Category', fontsize=14)
        plt.ylabel('Number of Articles', fontsize=14)
        plt.grid(axis='y', alpha=0.3)

        # Add percentage annotations
        total = len(self.sentiment_df)
        for i, (label, count) in enumerate(sentiment_counts.items()):
            percentage = (count / total) * 100
            plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=14, color='white')

        # Add summary statistics
        plt.text(0.02, 0.98, f'Total Articles: {total}\\nAnalysis Period: Sep-Oct 2025',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_distribution_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/sentiment_distribution_real.png")

    def create_company_sentiment_viz(self):
        """Create company sentiment analysis with real company names"""
        print("Creating company sentiment visualization...")

        plt.figure(figsize=(14, 10))

        # Calculate sentiment by company
        company_stats = self.sentiment_df.groupby('company').agg({
            'sentiment_score': ['mean', 'count', 'std']
        }).round(3)

        company_stats.columns = ['mean_sentiment', 'count', 'std_sentiment']
        company_stats = company_stats.sort_values('mean_sentiment')

        # Create horizontal bar chart
        colors = ['red' if x < -0.1 else 'green' if x > 0.1 else 'orange'
                 for x in company_stats['mean_sentiment']]

        bars = plt.barh(company_stats.index, company_stats['mean_sentiment'],
                       xerr=company_stats['std_sentiment'], capsize=5,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # Add count labels
        for i, (company, row) in enumerate(company_stats.iterrows()):
            count = int(row['count'])
            sentiment = row['mean_sentiment']
            plt.text(sentiment + (0.02 if sentiment >= 0 else -0.02), i,
                    f'n={count}', va='center',
                    ha='left' if sentiment >= 0 else 'right',
                    fontweight='bold', fontsize=11)

        plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        plt.title('Figure 4.2D: Average Sentiment by GSE Company',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Average Sentiment Score (-1 to +1)', fontsize=14)
        plt.ylabel('Company', fontsize=14)
        plt.grid(axis='x', alpha=0.3)

        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Negative (< -0.1)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Neutral (-0.1 to 0.1)'),
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Positive (> 0.1)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        # Add statistics box
        stats_text = f"Total Companies: {len(company_stats)}\\nMost Positive: {company_stats['mean_sentiment'].idxmax()}\\nMost Negative: {company_stats['mean_sentiment'].idxmin()}"
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/company_sentiment_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/company_sentiment_real.png")

    def create_source_sentiment_viz(self):
        """Create news source sentiment analysis"""
        print("Creating news source sentiment visualization...")

        plt.figure(figsize=(12, 8))

        # Calculate sentiment by source
        source_stats = self.sentiment_df.groupby('source').agg({
            'sentiment_score': ['mean', 'count', 'std']
        }).round(3)

        source_stats.columns = ['mean_sentiment', 'count', 'std_sentiment']
        source_stats = source_stats.sort_values('mean_sentiment', ascending=False)

        # Create horizontal bar chart
        colors = ['darkblue' if x > 0 else 'darkred' for x in source_stats['mean_sentiment']]

        bars = plt.barh(source_stats.index, source_stats['mean_sentiment'],
                       xerr=source_stats['std_sentiment'], capsize=4,
                       color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        # Add count labels
        for i, (source, row) in enumerate(source_stats.iterrows()):
            count = int(row['count'])
            sentiment = row['mean_sentiment']
            plt.text(sentiment + (0.01 if sentiment >= 0 else -0.01), i,
                    f'n={count}', va='center',
                    ha='left' if sentiment >= 0 else 'right',
                    fontweight='bold', fontsize=10)

        plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        plt.title('Figure 4.2C: Average Sentiment by News Source',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Average Sentiment Score (-1 to +1)', fontsize=14)
        plt.ylabel('News Source', fontsize=14)
        plt.grid(axis='x', alpha=0.3)

        # Add statistics
        plt.text(0.02, 0.98,
                f"Sources Analyzed: {len(source_stats)}\\nMost Positive: {source_stats['mean_sentiment'].idxmax()}\\nMost Negative: {source_stats['mean_sentiment'].idxmin()}",
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/source_sentiment_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/source_sentiment_real.png")

    def create_sentiment_timeline_viz(self):
        """Create sentiment timeline visualization"""
        print("Creating sentiment timeline visualization...")

        plt.figure(figsize=(16, 8))

        # Aggregate sentiment by date
        daily_sentiment = self.sentiment_df.groupby('date').agg({
            'sentiment_score': ['mean', 'count', 'std']
        }).dropna()

        daily_sentiment.columns = ['mean_sentiment', 'count', 'std_sentiment']

        # Create the plot
        plt.plot(daily_sentiment.index, daily_sentiment['mean_sentiment'],
                marker='o', markersize=6, linewidth=2, color='#1f77b4',
                markerfacecolor='white', markeredgecolor='#1f77b4', markeredgewidth=2)

        # Add confidence bands
        plt.fill_between(daily_sentiment.index,
                        daily_sentiment['mean_sentiment'] - daily_sentiment['std_sentiment'],
                        daily_sentiment['mean_sentiment'] + daily_sentiment['std_sentiment'],
                        alpha=0.2, color='#1f77b4', label='±1 Std Dev')

        # Add zero line
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Neutral (0)')

        plt.title('Figure 4.2B: Daily Sentiment Trend Analysis',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Average Sentiment Score', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')

        # Add statistics annotation
        mean_sentiment = daily_sentiment['mean_sentiment'].mean()
        std_sentiment = daily_sentiment['mean_sentiment'].std()
        days_analyzed = len(daily_sentiment)

        stats_text = f"""Analysis Period: {daily_sentiment.index.min().strftime('%b %d')} - {daily_sentiment.index.max().strftime('%b %d, %Y')}
Days with Data: {days_analyzed}
Overall Mean: {mean_sentiment:.3f}
Overall Std: {std_sentiment:.3f}
Max Sentiment: {daily_sentiment['mean_sentiment'].max():.3f}
Min Sentiment: {daily_sentiment['mean_sentiment'].min():.3f}"""

        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
                fontsize=10, family='monospace')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_timeline_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/sentiment_timeline_real.png")

    def create_stock_price_viz(self):
        """Create stock price evolution visualization"""
        print("Creating stock price visualization...")

        if self.stock_df.empty or len(self.stock_df) == 0:
            print("No stock data available for visualization")
            return

        plt.figure(figsize=(16, 8))

        # Plot price over time
        plt.plot(self.stock_df['Date'], self.stock_df['Close'],
                linewidth=3, color='#2ca02c', alpha=0.8, label='GSE Composite Index')

        # Add moving averages if data allows
        if len(self.stock_df) > 10:
            ma5 = self.stock_df['Close'].rolling(window=5).mean()
            ma10 = self.stock_df['Close'].rolling(window=10).mean()

            plt.plot(self.stock_df['Date'], ma5, linewidth=2, color='orange',
                    label='5-Day Moving Average', alpha=0.8)
            plt.plot(self.stock_df['Date'], ma10, linewidth=2, color='red',
                    label='10-Day Moving Average', alpha=0.8)

        plt.title('Figure 4.3A: GSE Composite Index Price Evolution',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Index Value (GHS)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper left')

        # Add price statistics
        if len(self.stock_df) > 0:
            price_range = f"{self.stock_df['Close'].min():.0f} - {self.stock_df['Close'].max():.0f}"
            avg_price = self.stock_df['Close'].mean()
            trading_days = len(self.stock_df)

            stats_text = f"""Trading Days: {trading_days}
Price Range: {price_range} GHS
Average Price: {avg_price:.0f} GHS
Latest Price: {self.stock_df['Close'].iloc[-1]:.0f} GHS"""

            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=11)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/stock_price_real.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/stock_price_real.png")

    def generate_all_proper_visualizations(self):
        """Generate all proper visualizations using real data"""
        print("Generating proper visualizations with real GSE data...")
        print("=" * 60)

        # Load real data
        self.load_real_data()

        # Generate sentiment visualizations
        self.create_sentiment_distribution_viz()
        self.create_sentiment_timeline_viz()
        self.create_source_sentiment_viz()
        self.create_company_sentiment_viz()

        # Generate stock visualizations
        self.create_stock_price_viz()

        print("\n" + "=" * 60)
        print("All proper visualizations generated!")
        print(f"Files saved in: {self.output_dir}/")

        # List generated files
        if os.path.exists(self.output_dir):
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
            print(f"\nGenerated {len(files)} visualization files:")
            for f in sorted(files):
                size = os.path.getsize(f'{self.output_dir}/{f}') / 1024
                print(f"  - {f} ({size:.1f} KB)")

        print("\nThese visualizations use REAL data from your GSE sentiment database!")
        print("Company names, sources, and actual sentiment scores are all genuine.")

def main():
    """Main function"""
    generator = GSEVisualizationGenerator()
    generator.generate_all_proper_visualizations()

if __name__ == "__main__":
    main()