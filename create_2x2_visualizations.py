#!/usr/bin/env python3
"""
Create 2x2 formatted visualizations for DOCX insertion
Supervisor wants two graphs per row instead of individual plots
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
plt.rcParams['figure.figsize'] = (16, 12)  # Larger figures for 2x2 layout
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

class GSE2x2VisualizationGenerator:
    """Generate 2x2 formatted visualizations for DOCX"""

    def __init__(self):
        self.db_path = "gse_sentiment.db"
        self.sentiment_df = None
        self.stock_df = None
        self.output_dir = "2x2_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_real_data(self):
        """Load real data from database and CSV files"""
        print("Loading real GSE data for 2x2 visualizations...")

        # Load sentiment data
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

        print(f"Loaded {len(self.sentiment_df)} sentiment entries")
        print(f"Companies: {sorted(self.sentiment_df['company'].unique())}")

    def create_sentiment_2x2_figure(self):
        """Create 2x2 sentiment analysis figure"""
        print("Creating 2x2 sentiment analysis figure...")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Sentiment distribution (top-left)
        sentiment_counts = self.sentiment_df['sentiment_label'].value_counts()
        colors = {'negative': '#e74c3c', 'positive': '#27ae60', 'neutral': '#f39c12'}
        bars = ax1.bar(sentiment_counts.index, sentiment_counts.values,
                      color=[colors.get(label, '#95a5a6') for label in sentiment_counts.index],
                      alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

        ax1.set_title('A) Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Sentiment Category', fontsize=14)
        ax1.set_ylabel('Number of Articles', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)

        # Add percentage annotations
        total = len(self.sentiment_df)
        for i, (label, count) in enumerate(sentiment_counts.items()):
            percentage = (count / total) * 100
            ax1.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=14, color='white')

        # 2. Daily sentiment trend (top-right)
        daily_sentiment = self.sentiment_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).dropna()
        daily_sentiment.columns = ['mean_sentiment', 'std_sentiment', 'count']

        ax2.plot(daily_sentiment.index, daily_sentiment['mean_sentiment'],
                marker='o', markersize=6, linewidth=3, color='#1f77b4',
                markerfacecolor='white', markeredgecolor='#1f77b4', markeredgewidth=2)

        # Add confidence bands
        ax2.fill_between(daily_sentiment.index,
                        daily_sentiment['mean_sentiment'] - daily_sentiment['std_sentiment'],
                        daily_sentiment['mean_sentiment'] + daily_sentiment['std_sentiment'],
                        alpha=0.2, color='#1f77b4', label='±1 Std Dev')

        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Neutral (0)')
        ax2.set_title('B) Daily Sentiment Trend', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('Date', fontsize=14)
        ax2.set_ylabel('Average Sentiment Score', fontsize=14)
        ax2.grid(alpha=0.3)
        ax2.legend(loc='upper right')

        # 3. Sentiment by source (bottom-left)
        source_sentiment = self.sentiment_df.groupby('source')['sentiment_score'].mean().sort_values()
        colors_source = ['darkblue' if x > 0 else 'darkred' for x in source_sentiment]

        bars_source = ax3.barh(source_sentiment.index, source_sentiment.values,
                              color=colors_source, alpha=0.7, edgecolor='black', linewidth=1)

        # Add count labels
        source_counts = self.sentiment_df['source'].value_counts()
        for i, (source, sentiment) in enumerate(source_sentiment.items()):
            count = source_counts.get(source, 0)
            ax3.text(sentiment + (0.01 if sentiment >= 0 else -0.01), i,
                    f'n={count}', va='center',
                    ha='left' if sentiment >= 0 else 'right',
                    fontweight='bold', fontsize=11)

        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        ax3.set_title('C) Sentiment by News Source', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('Average Sentiment Score (-1 to +1)', fontsize=14)
        ax3.set_ylabel('News Source', fontsize=14)
        ax3.grid(axis='x', alpha=0.3)

        # 4. Sentiment by company (bottom-right)
        company_sentiment = self.sentiment_df.groupby('company')['sentiment_score'].mean().sort_values()
        colors_company = ['red' if x < -0.1 else 'green' if x > 0.1 else 'orange'
                         for x in company_sentiment]

        bars_company = ax4.barh(company_sentiment.index, company_sentiment.values,
                               color=colors_company, alpha=0.7, edgecolor='black', linewidth=1)

        # Add count labels
        company_counts = self.sentiment_df['company'].value_counts()
        for i, (company, sentiment) in enumerate(company_sentiment.items()):
            count = company_counts.get(company, 0)
            ax4.text(sentiment + (0.02 if sentiment >= 0 else -0.02), i,
                    f'n={count}', va='center',
                    ha='left' if sentiment >= 0 else 'right',
                    fontweight='bold', fontsize=11)

        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        ax4.set_title('D) Sentiment by GSE Company', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('Average Sentiment Score (-1 to +1)', fontsize=14)
        ax4.set_ylabel('Company', fontsize=14)
        ax4.grid(axis='x', alpha=0.3)

        # Add legend for company plot
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Negative (< -0.1)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Neutral (-0.1 to 0.1)'),
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Positive (> 0.1)')
        ]
        ax4.legend(handles=legend_elements, loc='lower right', fontsize=12)

        # Overall title and statistics
        fig.suptitle('Figure 4.2: GSE Sentiment Analysis Overview', fontsize=20, fontweight='bold', y=0.98)

        # Add summary statistics as text
        stats_text = f"""Analysis Summary:
• Total Articles: {len(self.sentiment_df)}
• Companies: {self.sentiment_df['company'].nunique()}
• Sources: {self.sentiment_df['source'].nunique()}
• Period: {self.sentiment_df['timestamp'].min().strftime('%b %d')} - {self.sentiment_df['timestamp'].max().strftime('%b %d, %Y')}
• Overall Sentiment: {self.sentiment_df['sentiment_score'].mean():.3f} ± {self.sentiment_df['sentiment_score'].std():.3f}"""

        fig.text(0.02, 0.02, stats_text, fontsize=12, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/figure_4_2_sentiment_2x2.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/figure_4_2_sentiment_2x2.png")

    def create_stock_market_2x2_figure(self):
        """Create 2x2 stock market analysis figure"""
        print("Creating 2x2 stock market analysis figure...")

        # Try to load stock data
        try:
            stock_df = pd.read_csv("GSE COMPOSITE INDEX.csv", header=None, skiprows=1)
            if len(stock_df) > 0:
                stock_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Turnover', 'Adj Close', 'Trades']
                stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%d/%m/%Y', errors='coerce')
                stock_df = stock_df.dropna(subset=['Date'])
                stock_df = stock_df.sort_values('Date')

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

                # 1. Price evolution (top-left)
                ax1.plot(stock_df['Date'], stock_df['Close'], linewidth=3, color='#2ca02c', alpha=0.8)
                ax1.set_title('A) GSE Composite Index Price Evolution', fontsize=16, fontweight='bold', pad=20)
                ax1.set_xlabel('Date', fontsize=14)
                ax1.set_ylabel('Index Value (GHS)', fontsize=14)
                ax1.grid(alpha=0.3)

                # 2. Daily returns distribution (top-right)
                returns = stock_df['Close'].pct_change().dropna()
                ax2.hist(returns, bins=50, color='green', alpha=0.7, edgecolor='black')
                ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
                ax2.axvline(x=returns.mean(), color='blue', linestyle='-', linewidth=2,
                           label=f'Mean: {returns.mean():.3f}')
                ax2.set_title('B) Distribution of Daily Returns', fontsize=16, fontweight='bold', pad=20)
                ax2.set_xlabel('Daily Return (%)', fontsize=14)
                ax2.set_ylabel('Frequency', fontsize=14)
                ax2.grid(alpha=0.3)
                ax2.legend()

                # 3. Trading volume (bottom-left)
                ax3.plot(stock_df['Date'], stock_df['Turnover'], linewidth=3, color='red', alpha=0.8)
                ax3.set_title('C) Daily Trading Volume', fontsize=16, fontweight='bold', pad=20)
                ax3.set_xlabel('Date', fontsize=14)
                ax3.set_ylabel('Volume (GHS)', fontsize=14)
                ax3.grid(alpha=0.3)

                # 4. Volume vs Price change (bottom-right)
                valid_data = stock_df.dropna(subset=['Turnover', 'Close'])
                if len(valid_data) > 1:
                    volume = valid_data['Turnover']
                    price_change = valid_data['Close'].pct_change()
                    ax4.scatter(volume, price_change, alpha=0.6, color='purple', edgecolors='black')
                    ax4.set_title('D) Volume vs Price Change Relationship', fontsize=16, fontweight='bold', pad=20)
                    ax4.set_xlabel('Trading Volume (GHS)', fontsize=14)
                    ax4.set_ylabel('Price Change (%)', fontsize=14)
                    ax4.grid(alpha=0.3)

                    # Add correlation
                    corr = volume.corr(price_change)
                    ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                            transform=ax4.transAxes, fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                fig.suptitle('Figure 4.3: GSE Stock Market Analysis', fontsize=20, fontweight='bold', y=0.98)

                # Add market statistics
                market_stats = f"""Market Statistics:
• Trading Days: {len(stock_df)}
• Price Range: {stock_df['Close'].min():.0f} - {stock_df['Close'].max():.0f} GHS
• Average Daily Volume: {stock_df['Turnover'].mean():,.0f} GHS
• Average Daily Return: {returns.mean():.3f}%
• Return Volatility: {returns.std():.3f}"""

                fig.text(0.02, 0.02, market_stats, fontsize=12, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/figure_4_3_stock_market_2x2.png', dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved: {self.output_dir}/figure_4_3_stock_market_2x2.png")
            else:
                print("No stock data available for 2x2 visualization")
        except Exception as e:
            print(f"Error creating stock market visualization: {e}")

    def generate_all_2x2_visualizations(self):
        """Generate all 2x2 formatted visualizations"""
        print("Generating 2x2 formatted visualizations for DOCX...")
        print("=" * 60)

        # Load real data
        self.load_real_data()

        # Generate 2x2 figures
        self.create_sentiment_2x2_figure()
        self.create_stock_market_2x2_figure()

        print("\n" + "=" * 60)
        print("2x2 VISUALIZATIONS GENERATED!")
        print("=" * 60)
        print("\nFiles created for DOCX insertion:")
        print("- figure_4_2_sentiment_2x2.png - 2x2 sentiment analysis")
        print("- figure_4_3_stock_market_2x2.png - 2x2 stock market analysis")

        # List generated files
        if os.path.exists(self.output_dir):
            files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
            print(f"\nGenerated {len(files)} 2x2 visualization files:")
            for f in sorted(files):
                size = os.path.getsize(f'{self.output_dir}/{f}') / 1024
                print(f"  - {f} ({size:.1f} KB)")

        print("\n" + "=" * 60)
        print("FORMATTING BENEFITS:")
        print("=" * 60)
        print("- Two graphs per row (2x2 layout)")
        print("- Larger, clearer individual plots")
        print("- Better readability for academic documents")
        print("- Professional formatting with proper labels")
        print("- Real GSE company names and data")
        print("- Supervisor-approved layout")

def main():
    """Main function"""
    generator = GSE2x2VisualizationGenerator()
    generator.generate_all_2x2_visualizations()

if __name__ == "__main__":
    main()