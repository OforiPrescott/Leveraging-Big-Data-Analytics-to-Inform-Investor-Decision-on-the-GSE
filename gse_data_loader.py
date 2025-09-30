"""
GSE Data Loader
Loads and processes Ghana Stock Exchange data from CSV files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class GSEDataLoader:
    """Load and process GSE stock data from CSV files"""
    
    def __init__(self):
        self.composite_data = None
        self.financial_data = None
        self.companies = None
        
    def load_gse_csv_data(self, composite_csv: str, financial_csv: str) -> bool:
        """Load GSE data from CSV files"""
        try:
            # Load GSE Composite Index data
            self.composite_data = self._process_gse_csv(composite_csv, 'GSE-CI')
            logger.info(f"Loaded {len(self.composite_data)} composite index records")
            
            # Load GSE Financial Index data
            self.financial_data = self._process_gse_csv(financial_csv, 'GSE-FSI')
            logger.info(f"Loaded {len(self.financial_data)} financial index records")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading GSE CSV data: {str(e)}")
            return False
    
    def _process_gse_csv(self, csv_file: str, index_type: str) -> pd.DataFrame:
        """Process GSE CSV file with proper parsing"""
        try:
            # Use pandas to read the CSV
            df = pd.read_csv(csv_file, header=None)

            # Find rows that contain actual data (look for date pattern in column 23)
            data_rows = []
            for idx, row in df.iterrows():
                # Check if column 23 (Date column) contains a date
                date_cell = str(row[23]) if len(row) > 23 else ""
                if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_cell.strip()):
                    # This looks like a data row - extract columns 23-30
                    data_row = row[23:31].values
                    data_rows.append(data_row)

            if not data_rows:
                raise ValueError("No valid data rows found in CSV file")

            # Create DataFrame from data rows
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Turnover', 'Turnover_Value', 'Trades']
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Clean and convert data types
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            
            # Clean numeric columns (remove quotes and commas)
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Turnover', 'Turnover_Value', 'Trades']
            for col in numeric_columns:
                df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid dates or prices
            df = df.dropna(subset=['Date', 'Close'])
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Add additional calculated columns
            df['Price_Change'] = df['Close'].diff()
            df['Price_Change_Pct'] = df['Close'].pct_change() * 100
            df['Index_Type'] = index_type
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {csv_file}: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # Moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            
            # RSI (Relative Strength Index)
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
            
            # Volume indicators
            df['Volume_MA'] = df['Turnover'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Turnover'] / df['Volume_MA']
            
            # Volatility
            df['Volatility'] = df['Price_Change_Pct'].rolling(window=10).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series([50] * len(prices), index=prices.index)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            ma = prices.rolling(window=window).mean()
            std = prices.rolling(window=window).std()
            upper_band = ma + (std * std_dev)
            lower_band = ma - (std * std_dev)
            return upper_band, lower_band
        except:
            return pd.Series([0] * len(prices), index=prices.index), pd.Series([0] * len(prices), index=prices.index)
    
    def get_stock_features(self, date: datetime, lookback_days: int = 30) -> Dict:
        """Get stock features for a specific date"""
        try:
            if self.composite_data is None:
                return self._get_default_stock_features()
            
            # Filter data up to the specified date
            end_date = pd.to_datetime(date)
            start_date = end_date - timedelta(days=lookback_days)
            
            recent_data = self.composite_data[
                (self.composite_data['Date'] <= end_date) & 
                (self.composite_data['Date'] >= start_date)
            ]
            
            if recent_data.empty:
                return self._get_default_stock_features()
            
            latest = recent_data.iloc[-1]
            
            features = {
                'close_price': float(latest['Close']),
                'price_change_1d': float(latest['Price_Change_Pct']) if pd.notna(latest['Price_Change_Pct']) else 0.0,
                'ma_5': float(latest['MA_5']) if pd.notna(latest['MA_5']) else float(latest['Close']),
                'ma_10': float(latest['MA_10']) if pd.notna(latest['MA_10']) else float(latest['Close']),
                'ma_20': float(latest['MA_20']) if pd.notna(latest['MA_20']) else float(latest['Close']),
                'rsi': float(latest['RSI']) if pd.notna(latest['RSI']) else 50.0,
                'volume': float(latest['Turnover']) if pd.notna(latest['Turnover']) else 0.0,
                'volume_ratio': float(latest['Volume_Ratio']) if pd.notna(latest['Volume_Ratio']) else 1.0,
                'volatility': float(latest['Volatility']) if pd.notna(latest['Volatility']) else 0.0,
                'bb_position': self._calculate_bb_position(latest) if pd.notna(latest['BB_Upper']) else 0.5
            }
            
            # Add price trend indicators
            if len(recent_data) >= 5:
                price_trend = recent_data['Close'].tail(5)
                features['price_trend_5d'] = float(np.polyfit(range(len(price_trend)), price_trend, 1)[0])
            else:
                features['price_trend_5d'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting stock features: {str(e)}")
            return self._get_default_stock_features()
    
    def _calculate_bb_position(self, row) -> float:
        """Calculate position within Bollinger Bands (0 = lower band, 1 = upper band)"""
        try:
            if pd.isna(row['BB_Upper']) or pd.isna(row['BB_Lower']) or row['BB_Upper'] == row['BB_Lower']:
                return 0.5
            
            position = (row['Close'] - row['BB_Lower']) / (row['BB_Upper'] - row['BB_Lower'])
            return max(0, min(1, position))
        except:
            return 0.5
    
    def _get_default_stock_features(self) -> Dict:
        """Return default stock features when data is not available"""
        return {
            'close_price': 100.0,
            'price_change_1d': 0.0,
            'ma_5': 100.0,
            'ma_10': 100.0,
            'ma_20': 100.0,
            'rsi': 50.0,
            'volume': 1000.0,
            'volume_ratio': 1.0,
            'volatility': 1.0,
            'bb_position': 0.5,
            'price_trend_5d': 0.0
        }
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of loaded data"""
        summary = {
            'composite_data': None,
            'financial_data': None
        }
        
        if self.composite_data is not None:
            summary['composite_data'] = {
                'total_records': len(self.composite_data),
                'date_range': {
                    'start': self.composite_data['Date'].min().strftime('%Y-%m-%d'),
                    'end': self.composite_data['Date'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': float(self.composite_data['Close'].min()),
                    'max': float(self.composite_data['Close'].max()),
                    'current': float(self.composite_data['Close'].iloc[-1])
                },
                'avg_daily_volume': float(self.composite_data['Turnover'].mean())
            }
        
        if self.financial_data is not None:
            summary['financial_data'] = {
                'total_records': len(self.financial_data),
                'date_range': {
                    'start': self.financial_data['Date'].min().strftime('%Y-%m-%d'),
                    'end': self.financial_data['Date'].max().strftime('%Y-%m-%d')
                },
                'price_range': {
                    'min': float(self.financial_data['Close'].min()),
                    'max': float(self.financial_data['Close'].max()),
                    'current': float(self.financial_data['Close'].iloc[-1])
                },
                'avg_daily_volume': float(self.financial_data['Turnover'].mean())
            }
        
        return summary
    
    def export_processed_data(self, output_path: str = "processed_gse_data.csv") -> bool:
        """Export processed data to CSV"""
        try:
            if self.composite_data is not None:
                self.composite_data.to_csv(f"composite_{output_path}", index=False)
                logger.info(f"Exported composite data to composite_{output_path}")
            
            if self.financial_data is not None:
                self.financial_data.to_csv(f"financial_{output_path}", index=False)
                logger.info(f"Exported financial data to financial_{output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data loader
    loader = GSEDataLoader()
    
    print("GSE Data Loader Test")
    print("=" * 30)
    
    # Test with sample data files (you would use your actual CSV files)
    try:
        # You would replace these with your actual file paths
        composite_csv = "GSE COMPOSITE INDEX.csv"
        financial_csv = "GSE FINANCIAL INDEX.csv"
        
        success = loader.load_gse_csv_data(composite_csv, financial_csv)
        
        if success:
            print("✅ Data loaded successfully!")
            
            # Get summary
            summary = loader.get_data_summary()
            print("\nData Summary:")
            print(f"Composite Index Records: {summary['composite_data']['total_records'] if summary['composite_data'] else 0}")
            print(f"Financial Index Records: {summary['financial_data']['total_records'] if summary['financial_data'] else 0}")
            
            # Test feature extraction
            features = loader.get_stock_features(datetime.now())
            print(f"\nSample Features:")
            for key, value in features.items():
                print(f"  {key}: {value:.3f}")
            
            # Export processed data
            loader.export_processed_data()
            
        else:
            print("❌ Failed to load data")
            
    except FileNotFoundError:
        print("⚠️  CSV files not found. This is expected in the demo.")
        print("Please ensure your GSE CSV files are in the same directory.")
        
        # Create sample data for testing
        print("\nCreating sample data for testing...")
        
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(5000, 7000, len(dates)),
            'High': np.random.uniform(5500, 7500, len(dates)),
            'Low': np.random.uniform(4500, 6500, len(dates)),
            'Close': np.random.uniform(5000, 7000, len(dates)),
            'Turnover': np.random.uniform(100000, 1000000, len(dates)),
            'Turnover_Value': np.random.uniform(1000000, 10000000, len(dates)),
            'Trades': np.random.uniform(100, 2000, len(dates))
        })
        
        # Add technical indicators
        sample_data['Price_Change'] = sample_data['Close'].diff()
        sample_data['Price_Change_Pct'] = sample_data['Close'].pct_change() * 100
        sample_data['MA_5'] = sample_data['Close'].rolling(5).mean()
        sample_data['MA_10'] = sample_data['Close'].rolling(10).mean()
        sample_data['RSI'] = 50  # Simplified
        
        loader.composite_data = sample_data
        
        print("✅ Sample data created!")
        features = loader.get_stock_features(datetime.now())
        print(f"\nSample Features from generated data:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")