#!/usr/bin/env python3
"""
Create individual visualization files for each subplot and analysis
This addresses the supervisor's feedback about visibility and separate explanations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import os
from datetime import datetime

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")

class IndividualVisualizationGenerator:
    """Generate individual visualization files for each analysis component"""

    def __init__(self, db_path="gse_sentiment.db"):
        self.db_path = db_path
        self.sentiment_df = None
        self.stock_df = None
        self.output_dir = "individual_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """Load sentiment and stock data"""
        print("Loading data for individual visualizations...")

        # Load sentiment data
        conn = sqlite3.connect(self.db_path)
        self.sentiment_df = pd.read_sql_query("""
            SELECT timestamp, source, sentiment_score, sentiment_label,
                   company, confidence, content
            FROM sentiment_data
        """, conn)
        conn.close()

        # Load stock data - handle variable column count
        temp_df = pd.read_csv("GSE COMPOSITE INDEX.csv", header=None)
        print(f"CSV loaded with shape: {temp_df.shape}")

        # Find rows that contain actual data (look for date pattern)
        data_rows = []
        for idx, row in temp_df.iterrows():
            # Check if any cell contains a date pattern
            date_found = False
            for cell in row:
                cell_str = str(cell)
                if pd.notna(cell) and ('/' in cell_str or '-' in cell_str):
                    try:
                        pd.to_datetime(cell_str, dayfirst=True, errors='coerce')
                        date_found = True
                        break
                    except:
                        continue
            if date_found:
                data_rows.append(row.values)

        if data_rows:
            # Create DataFrame from data rows
            self.stock_df = pd.DataFrame(data_rows)
            # Use only first 8 columns or available columns
            n_cols = min(8, len(self.stock_df.columns))
            self.stock_df = self.stock_df.iloc[:, :n_cols]
            self.stock_df.columns = ['Date', 'Open', 'High', 'Low', 'Close',
                                    'Turnover', 'Adj Close', 'Trades'][:n_cols]
        else:
            # Fallback: use first 8 columns
            self.stock_df = temp_df.iloc[:, :8]
            self.stock_df.columns = ['Date', 'Open', 'High', 'Low', 'Close',
                                    'Turnover', 'Adj Close', 'Trades']

        # Process data
        self._process_data()

        print(f"Loaded {len(self.sentiment_df)} sentiment entries and {len(self.stock_df)} stock records")

    def _process_data(self):
        """Process and clean the data"""
        # Process sentiment data
        self.sentiment_df['timestamp'] = pd.to_datetime(self.sentiment_df['timestamp'], errors='coerce')
        self.sentiment_df['date'] = self.sentiment_df['timestamp'].dt.date
        self.sentiment_df['date'] = pd.to_datetime(self.sentiment_df['date'])

        # Process stock data
        self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date'], format='%d/%m/%Y', errors='coerce')
        self.stock_df = self.stock_df.dropna(subset=['Date'])

        # Add technical indicators
        self.stock_df['Price_Change'] = self.stock_df['Close'].pct_change()
        self.stock_df['MA_5'] = self.stock_df['Close'].rolling(window=5).mean()
        self.stock_df['MA_10'] = self.stock_df['Close'].rolling(window=10).mean()
        self.stock_df['Price_Change_1d'] = self.stock_df['Close'].pct_change(1)
        self.stock_df['Price_Change_5d'] = self.stock_df['Close'].pct_change(5)
        self.stock_df['Volume_MA'] = self.stock_df['Turnover'].rolling(window=10).mean()
        self.stock_df['Volume_Ratio'] = self.stock_df['Turnover'] / self.stock_df['Volume_MA']

    def create_sentiment_distribution_plot(self):
        """Create individual sentiment distribution plot"""
        print("Creating sentiment distribution plot...")

        plt.figure(figsize=(10, 6))

        # Create the bar chart
        sentiment_counts = self.sentiment_df['sentiment_label'].value_counts()
        colors = ['red', 'blue', 'green']  # negative, positive, neutral

        bars = plt.bar(sentiment_counts.index, sentiment_counts.values,
                      color=colors[:len(sentiment_counts)], alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')

        plt.title('Figure 4.2A: Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Number of Entries')
        plt.grid(axis='y', alpha=0.3)

        # Add percentage annotations
        total = len(self.sentiment_df)
        for i, (label, count) in enumerate(sentiment_counts.items()):
            percentage = (count / total) * 100
            plt.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center',
                    fontweight='bold', fontsize=12, color='white')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/sentiment_distribution.png")

    def create_daily_sentiment_trend_plot(self):
        """Create individual daily sentiment trend plot"""
        print("Creating daily sentiment trend plot...")

        plt.figure(figsize=(12, 6))

        # Aggregate sentiment by date
        daily_sentiment = self.sentiment_df.groupby('date')['sentiment_score'].agg(['mean', 'count'])

        # Create the line plot
        plt.plot(daily_sentiment.index, daily_sentiment['mean'],
                marker='o', linewidth=2, markersize=4, color='#1f77b4')

        # Add zero line
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Neutral (0)')

        plt.title('Figure 4.2B: Daily Sentiment Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.grid(alpha=0.3)
        plt.legend()

        # Add statistics annotation
        mean_sentiment = daily_sentiment['mean'].mean()
        std_sentiment = daily_sentiment['mean'].std()
        plt.text(0.02, 0.98, f'Mean: {mean_sentiment:.3f}\\nStd: {std_sentiment:.3f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/daily_sentiment_trend.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/daily_sentiment_trend.png")

    def create_sentiment_by_source_plot(self):
        """Create individual sentiment by source plot"""
        print("Creating sentiment by source plot...")

        plt.figure(figsize=(12, 6))

        # Calculate average sentiment by source
        source_sentiment = self.sentiment_df.groupby('source')['sentiment_score'].agg(['mean', 'count', 'std'])

        # Create horizontal bar chart
        bars = plt.barh(source_sentiment.index, source_sentiment['mean'],
                       xerr=source_sentiment['std'], capsize=5, alpha=0.7, color='#ff7f0e')

        # Add count labels
        for i, (idx, row) in enumerate(source_sentiment.iterrows()):
            plt.text(row['mean'] + 0.01, i, f'n={int(row["count"])}',
                    va='center', fontweight='bold')

        plt.title('Figure 4.2C: Sentiment by News Source', fontsize=14, fontweight='bold')
        plt.xlabel('Average Sentiment Score')
        plt.ylabel('News Source')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_by_source.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/sentiment_by_source.png")

    def create_sentiment_by_company_plot(self):
        """Create individual sentiment by company plot"""
        print("Creating sentiment by company plot...")

        plt.figure(figsize=(12, 8))

        # Calculate sentiment by company
        company_sentiment = self.sentiment_df.groupby('company')['sentiment_score'].agg(['mean', 'count', 'std'])

        # Sort by mean sentiment
        company_sentiment = company_sentiment.sort_values('mean', ascending=True)

        # Create horizontal bar chart
        colors = ['red' if x < 0 else 'green' for x in company_sentiment['mean']]
        bars = plt.barh(company_sentiment.index, company_sentiment['mean'],
                       xerr=company_sentiment['std'], capsize=3, alpha=0.7, color=colors)

        # Add count labels
        for i, (idx, row) in enumerate(company_sentiment.iterrows()):
            plt.text(row['mean'] + (0.02 if row['mean'] >= 0 else -0.02), i,
                    f'n={int(row["count"])}', va='center',
                    ha='left' if row['mean'] >= 0 else 'right', fontweight='bold')

        plt.title('Figure 4.2D: Sentiment by Company', fontsize=14, fontweight='bold')
        plt.xlabel('Average Sentiment Score')
        plt.ylabel('Company')
        plt.grid(axis='x', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Neutral')

        # Add legend
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_by_company.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/sentiment_by_company.png")

    def create_price_evolution_plot(self):
        """Create individual GSE price evolution plot"""
        print("Creating GSE price evolution plot...")

        plt.figure(figsize=(14, 6))

        plt.plot(self.stock_df['Date'], self.stock_df['Close'],
                linewidth=2, color='#1f77b4', alpha=0.8)

        # Add moving averages
        plt.plot(self.stock_df['Date'], self.stock_df['MA_5'],
                linewidth=1.5, color='orange', label='5-day MA', alpha=0.8)
        plt.plot(self.stock_df['Date'], self.stock_df['MA_10'],
                linewidth=1.5, color='red', label='10-day MA', alpha=0.8)

        plt.title('Figure 4.3A: GSE Composite Index Price Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Index Value (GHS)')
        plt.grid(alpha=0.3)
        plt.legend()

        # Add statistics
        price_range = f"{self.stock_df['Close'].min():.0f} - {self.stock_df['Close'].max():.0f}"
        avg_change = self.stock_df['Price_Change'].mean() * 100
        plt.text(0.02, 0.98, f'Price Range: {price_range} GHS\\nAvg Daily Change: {avg_change:.2f}%',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/gse_price_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/gse_price_evolution.png")

    def create_returns_distribution_plot(self):
        """Create individual returns distribution plot"""
        print("Creating returns distribution plot...")

        plt.figure(figsize=(10, 6))

        # Filter out extreme outliers for better visualization
        returns = self.stock_df['Price_Change'].dropna()
        returns_filtered = returns[(returns > returns.quantile(0.01)) &
                                  (returns < returns.quantile(0.99))]

        plt.hist(returns_filtered * 100, bins=50, alpha=0.7, color='#2ca02c', edgecolor='black')

        # Add vertical lines for mean and median
        mean_return = returns.mean() * 100
        median_return = returns.median() * 100

        plt.axvline(mean_return, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_return:.2f}%')
        plt.axvline(median_return, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_return:.2f}%')

        plt.title('Figure 4.3B: Daily Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return (%)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.legend()

        # Add statistics
        std_return = returns.std() * 100
        plt.text(0.02, 0.98, f'Std Dev: {std_return:.2f}%\\nSkewness: {returns.skew():.3f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/returns_distribution.png")

    def create_trading_volume_plot(self):
        """Create individual trading volume plot"""
        print("Creating trading volume plot...")

        plt.figure(figsize=(14, 6))

        plt.plot(self.stock_df['Date'], self.stock_df['Turnover'] / 1000000,  # Convert to millions
                linewidth=2, color='#d62728', alpha=0.8)

        # Add volume moving average
        volume_ma = self.stock_df['Turnover'].rolling(window=30).mean() / 1000000
        plt.plot(self.stock_df['Date'], volume_ma,
                linewidth=2, color='orange', label='30-day MA', alpha=0.8)

        plt.title('Figure 4.3C: Daily Trading Volume', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Trading Volume (Millions GHS)')
        plt.grid(alpha=0.3)
        plt.legend()

        # Add statistics
        avg_volume = self.stock_df['Turnover'].mean() / 1000000
        max_volume = self.stock_df['Turnover'].max() / 1000000
        plt.text(0.02, 0.98, f'Avg Volume: {avg_volume:.1f}M GHS\\nMax Volume: {max_volume:.1f}M GHS',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/trading_volume.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/trading_volume.png")

    def create_day_of_week_returns_plot(self):
        """Create individual day of week returns plot"""
        print("Creating day of week returns plot...")

        plt.figure(figsize=(10, 6))

        # Add day of week
        self.stock_df['Day_of_Week'] = self.stock_df['Date'].dt.day_name()

        # Calculate average returns by day
        day_returns = self.stock_df.groupby('Day_of_Week')['Price_Change'].mean() * 100
        day_counts = self.stock_df.groupby('Day_of_Week')['Price_Change'].count()

        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        day_returns = day_returns.reindex(day_order)
        day_counts = day_counts.reindex(day_order)

        # Create bar chart
        bars = plt.bar(day_returns.index, day_returns.values, alpha=0.7, color='#9467bd')

        # Color bars based on performance
        for i, bar in enumerate(bars):
            if day_returns.iloc[i] > 0:
                bar.set_color('#2ca02c')  # green for positive
            else:
                bar.set_color('#d62728')  # red for negative

        # Add count labels
        for i, (day, count) in enumerate(day_counts.items()):
            plt.text(i, day_returns.iloc[i] + (0.01 if day_returns.iloc[i] >= 0 else -0.01),
                    f'n={count}', ha='center',
                    va='bottom' if day_returns.iloc[i] >= 0 else 'top', fontweight='bold')

        plt.title('Figure 4.3D: Average Returns by Day of Week', fontsize=14, fontweight='bold')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Daily Return (%)')
        plt.grid(axis='y', alpha=0.3)

        # Add overall average line
        overall_avg = day_returns.mean()
        plt.axhline(y=overall_avg, color='black', linestyle='--', alpha=0.7,
                   label=f'Overall Avg: {overall_avg:.2f}%')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/day_of_week_returns.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {self.output_dir}/day_of_week_returns.png")

    def create_correlation_heatmap(self):
        """Create individual correlation heatmap"""
        print("Creating correlation heatmap...")

        # Merge datasets for correlation analysis
        self._merge_datasets()

        if hasattr(self, 'merged_df') and self.merged_df is not None:
            # Select numeric columns for correlation
            numeric_cols = ['sentiment_score', 'confidence', 'Close', 'Price_Change',
                          'Turnover', 'MA_5', 'MA_10', 'Price_Change_1d', 'Price_Change_5d',
                          'Volume_Ratio']

            # Filter to available columns
            available_cols = [col for col in numeric_cols if col in self.merged_df.columns]
            corr_data = self.merged_df[available_cols].dropna()

            if len(corr_data) > 0:
                plt.figure(figsize=(12, 10))

                # Create correlation matrix
                corr_matrix = corr_data.corr()

                # Create heatmap
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                           center=0, fmt='.2f', square=True, linewidths=0.5,
                           cbar_kws={"shrink": 0.8})

                plt.title('Figure 4.4: Feature Correlation Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()

                plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved: {self.output_dir}/correlation_heatmap.png")
            else:
                print("No correlation data available")
        else:
            print("Merged dataset not available for correlation analysis")

    def _merge_datasets(self):
        """Merge sentiment and stock data for correlation analysis"""
        if self.sentiment_df is None or self.stock_df is None:
            return

        # Aggregate sentiment by date
        sentiment_daily = self.sentiment_df.groupby('date').agg({
            'sentiment_score': 'mean',
            'confidence': 'mean'
        }).reset_index()

        # Merge with stock data
        self.merged_df = pd.merge(
            self.stock_df,
            sentiment_daily,
            left_on='Date',
            right_on='date',
            how='left'
        )

    def generate_all_visualizations(self):
        """Generate all individual visualizations"""
        print("Generating all individual visualizations...")
        print("=" * 50)

        # Load data first
        self.load_data()

        # Generate sentiment plots
        self.create_sentiment_distribution_plot()
        self.create_daily_sentiment_trend_plot()
        self.create_sentiment_by_source_plot()
        self.create_sentiment_by_company_plot()

        # Generate stock market plots
        self.create_price_evolution_plot()
        self.create_returns_distribution_plot()
        self.create_trading_volume_plot()
        self.create_day_of_week_returns_plot()

        # Generate correlation plot
        self.create_correlation_heatmap()

        print("\n" + "=" * 50)
        print("All individual visualizations generated!")
        print(f"Files saved in: {self.output_dir}/")
        print("\nGenerated files:")
        visualizations = os.listdir(self.output_dir)
        for viz in sorted(visualizations):
            if viz.endswith('.png'):
                print(f"  - {viz}")

def main():
    """Main function to generate all individual visualizations"""
    generator = IndividualVisualizationGenerator()
    generator.generate_all_visualizations()

if __name__ == "__main__":
    main()