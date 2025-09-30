"""
Setup and Run Script for GSE Sentiment Analysis System
Complete setup, data collection, and system initialization
"""

import os
import sys
import subprocess
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gse_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GSESystemSetup:
    """Complete setup for GSE Sentiment Analysis System"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
    
    def install_requirements(self):
        """Install required Python packages"""
        logger.info("Installing required packages...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            logger.info("[SUCCESS] Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Error installing requirements: {e}")
            return False
    
    def setup_databases(self):
        """Setup SQLite databases"""
        logger.info("Setting up databases...")
        
        try:
            # Main sentiment database
            conn = sqlite3.connect(self.data_dir / "gse_sentiment.db")
            cursor = conn.cursor()
            
            # Sentiment data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    source TEXT,
                    content TEXT,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    company TEXT,
                    url TEXT,
                    confidence REAL,
                    content_hash TEXT UNIQUE
                )
            ''')
            
            # Manual sentiment table
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
            
            # Stock data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    symbol TEXT,
                    open_price REAL,
                    high_price REAL,
                    low_price REAL,
                    close_price REAL,
                    volume INTEGER,
                    price_change REAL,
                    percent_change REAL
                )
            ''')
            
            # News articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    date_published DATETIME,
                    scraped_at DATETIME,
                    source TEXT,
                    relevance_score REAL,
                    content_hash TEXT
                )
            ''')
            
            # Social media posts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS social_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    created_at DATETIME,
                    scraped_at DATETIME,
                    url TEXT,
                    score INTEGER,
                    relevance_score REAL,
                    content_hash TEXT UNIQUE
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    company TEXT,
                    training_date DATETIME,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    parameters TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_company ON sentiment_data(company)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment_data(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_manual_company ON manual_sentiment(company)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_manual_timestamp ON manual_sentiment(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stock_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(date)')
            
            conn.commit()
            conn.close()
            
            logger.info("[SUCCESS] Database setup completed")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error setting up databases: {e}")
            return False
    
    def load_gse_csv_data(self):
        """Load GSE CSV data if available"""
        logger.info("Loading GSE CSV data...")
        
        try:
            csv_files = {
                'composite': 'GSE COMPOSITE INDEX.csv',
                'financial': 'GSE FINANCIAL INDEX.csv'
            }
            
            loaded_files = {}
            
            for file_type, filename in csv_files.items():
                file_path = self.base_dir / filename
                if file_path.exists():
                    logger.info(f"Loading {filename}...")
                    df = self._process_gse_csv(file_path)
                    if df is not None:
                        loaded_files[file_type] = df
                        logger.info(f"[SUCCESS] Loaded {len(df)} records from {filename}")
                else:
                    logger.warning(f"[WARNING]  {filename} not found, creating sample data...")
                    loaded_files[file_type] = self._create_sample_stock_data(file_type)
            
            # Save to database
            if loaded_files:
                self._save_stock_data_to_db(loaded_files)
                logger.info("[SUCCESS] Stock data saved to database")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading CSV data: {e}")
            return False
    
    def _process_gse_csv(self, file_path: Path) -> pd.DataFrame:
        """Process GSE CSV file"""
        try:
            # Read CSV with proper parsing
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Extract data rows
            data_rows = []
            for line in lines:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 8:
                    try:
                        # Check if first part looks like a date
                        date_str = parts[0]
                        if '/' in date_str and len(date_str.split('/')) == 3:
                            data_rows.append(parts)
                    except:
                        continue
            
            if not data_rows:
                return None
            
            # Create DataFrame
            columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Turnover', 'Turnover_Value', 'Trades']
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Clean and convert data
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Turnover', 'Turnover_Value', 'Trades']
            for col in numeric_columns:
                df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove invalid rows
            df = df.dropna(subset=['Date', 'Close'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def _create_sample_stock_data(self, data_type: str) -> pd.DataFrame:
        """Create sample stock data for testing"""
        logger.info(f"Creating sample {data_type} data...")
        
        # Generate 2 years of daily data
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Remove weekends (basic approach)
        dates = [d for d in dates if d.weekday() < 5]
        
        # Base price ranges
        if data_type == 'composite':
            base_price = 6000
            volatility = 200
        else:  # financial
            base_price = 3500
            volatility = 150
        
        # Generate random walk price data
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Small positive drift
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))  # Floor at 50% of base
        
        # Create OHLC data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate OHLC around close price
            volatility_factor = np.random.uniform(0.5, 1.5)
            daily_vol = volatility * volatility_factor / 100
            
            high = close * (1 + abs(np.random.normal(0, daily_vol)))
            low = close * (1 - abs(np.random.normal(0, daily_vol)))
            open_price = low + (high - low) * np.random.uniform(0.2, 0.8)
            
            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            base_volume = 500000 if data_type == 'composite' else 200000
            volume = int(base_volume * np.random.uniform(0.3, 3.0))
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(close, 2),
                'Turnover': volume,
                'Turnover_Value': round(volume * close, 2),
                'Trades': int(volume / np.random.uniform(100, 500))
            })
        
        df = pd.DataFrame(data)
        logger.info(f"[SUCCESS] Created {len(df)} sample records for {data_type} data")
        return df
    
    def _save_stock_data_to_db(self, loaded_files: dict):
        """Save stock data to database"""
        try:
            conn = sqlite3.connect(self.data_dir / "gse_sentiment.db")
            
            for file_type, df in loaded_files.items():
                symbol = 'GSE-CI' if file_type == 'composite' else 'GSE-FSI'
                
                for _, row in df.iterrows():
                    # Calculate price change
                    price_change = 0.0
                    percent_change = 0.0
                    
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR REPLACE INTO stock_data 
                            (date, symbol, open_price, high_price, low_price, close_price, 
                             volume, price_change, percent_change)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            row['Date'].date(),
                            symbol,
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            int(row['Turnover']),
                            price_change,
                            percent_change
                        ))
                    except Exception as e:
                        logger.warning(f"Error inserting row: {e}")
                        continue
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving stock data: {e}")
    
    def create_sample_sentiment_data(self):
        """Create sample sentiment data for testing"""
        logger.info("Creating sample sentiment data...")
        
        try:
            from gse_sentiment_analysis_system import GSESentimentAnalyzer
            
            analyzer = GSESentimentAnalyzer(db_path=str(self.data_dir / "gse_sentiment.db"))
            
            # Sample news content for different companies
            sample_news = [
                {
                    'company': 'MTN',
                    'content': 'MTN Ghana reports strong quarterly earnings with 15% growth in subscriber base',
                    'sentiment': 'positive',
                    'source': 'business_ghana'
                },
                {
                    'company': 'GCB',
                    'content': 'GCB Bank announces new digital banking platform to compete with fintech companies',
                    'sentiment': 'positive',
                    'source': 'ghanaweb'
                },
                {
                    'company': 'TOTAL',
                    'content': 'TotalEnergies Ghana faces regulatory challenges over fuel pricing',
                    'sentiment': 'negative',
                    'source': 'graphic_online'
                },
                {
                    'company': 'EGH',
                    'content': 'Ecobank Ghana maintains stable performance despite economic challenges',
                    'sentiment': 'neutral',
                    'source': 'myjoyonline'
                },
                {
                    'company': 'AGA',
                    'content': 'AngloGold Ashanti reports record gold production in Q3',
                    'sentiment': 'very_positive',
                    'source': 'reuters'
                }
            ]
            
            # Create sentiment data entries
            sentiment_entries = []
            for i in range(50):  # Create 50 sample entries
                news = np.random.choice(sample_news)
                
                # Add some variation to the content
                variations = [
                    f"Breaking: {news['content']}",
                    f"Market Update: {news['content']}",
                    f"Ghana Business News: {news['content']}",
                    news['content']
                ]
                
                content = np.random.choice(variations)
                
                # Generate timestamp within last 30 days
                timestamp = datetime.now() - timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                # Convert sentiment to score
                sentiment_scores = {
                    'very_positive': 0.8,
                    'positive': 0.4,
                    'neutral': 0.0,
                    'negative': -0.4,
                    'very_negative': -0.8
                }
                
                from gse_sentiment_analysis_system import SentimentData
                
                entry = SentimentData(
                    timestamp=timestamp,
                    source=news['source'],
                    content=content,
                    sentiment_score=sentiment_scores[news['sentiment']],
                    sentiment_label=news['sentiment'],
                    company=news['company'],
                    url=f"https://example.com/news/{i}",
                    confidence=np.random.uniform(0.6, 0.9)
                )
                
                sentiment_entries.append(entry)
            
            # Save to database
            analyzer.save_sentiment_data(sentiment_entries)
            logger.info(f"[SUCCESS] Created {len(sentiment_entries)} sample sentiment entries")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error creating sample sentiment data: {e}")
            return False
    
    def create_configuration_file(self):
        """Create configuration file for the system, preserving existing API keys"""
        logger.info("Creating configuration file...")

        # Default configuration
        default_config = {
            "database": {
                "path": str(self.data_dir / "gse_sentiment.db"),
                "backup_interval_hours": 24
            },
            "data_collection": {
                "news_sources": [
                    "ghanaweb",
                    "myjoyonline",
                    "graphic_online",
                    "business_ghana",
                    "citi_newsroom"
                ],
                "social_platforms": [
                    "reddit",
                    "linkedin",
                    "twitter"
                ],
                "collection_interval_hours": 6,
                "max_articles_per_source": 20
            },
            "sentiment_analysis": {
                "models": ["vader", "textblob", "bert"],
                "confidence_threshold": 0.5,
                "relevance_threshold": 0.1
            },
            "prediction_models": {
                "algorithms": ["random_forest", "gradient_boosting", "logistic_regression"],
                "retrain_interval_days": 7,
                "feature_window_days": 30
            },
            "dashboard": {
                "port": 8501,
                "refresh_interval_minutes": 30
            },
            "logging": {
                "level": "INFO",
                "log_file": str(self.logs_dir / "gse_system.log"),
                "max_log_size_mb": 10
            },
            "api_keys": {
                "facebook": {
                    "app_id": "",
                    "app_secret": "",
                    "access_token": "",
                    "page_access_token": ""
                },
                "linkedin": {
                    "client_id": "",
                    "client_secret": "",
                    "access_token": "",
                    "refresh_token": ""
                },
                "twitter": {
                    "api_key": "",
                    "api_secret": "",
                    "access_token": "",
                    "access_token_secret": "",
                    "bearer_token": ""
                }
            }
        }

        config_path = self.base_dir / "config.json"

        # Check if config.json already exists and preserve API keys
        existing_api_keys = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    existing_config = json.load(f)
                    existing_api_keys = existing_config.get('api_keys', {})
                logger.info("Found existing config.json, preserving API keys")
            except Exception as e:
                logger.warning(f"Could not read existing config.json: {e}")

        # Merge existing API keys with defaults
        if existing_api_keys:
            default_config['api_keys'] = existing_api_keys
            logger.info("[SUCCESS] Preserved existing API keys in configuration")

        # Write the configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"[SUCCESS] Configuration file created/updated at {config_path}")
        return True
    
    def run_initial_data_collection(self):
        """Run initial data collection"""
        logger.info("Running initial data collection...")
        
        try:
            # Import and run scrapers
            from news_scraper import NewsScraper
            from social_media_scraper import SocialMediaScraper
            
            # News scraping
            news_scraper = NewsScraper()
            search_terms = ["Ghana Stock Exchange", "GSE trading", "Ghana financial news"]
            
            for term in search_terms:
                logger.info(f"Collecting news for: {term}")
                articles = news_scraper.scrape_multiple_sources(term, max_articles_per_source=5)
                if articles:
                    news_scraper.save_articles_to_db(articles, str(self.data_dir / "news_articles.db"))
                    logger.info(f"[SUCCESS] Collected {len(articles)} news articles for '{term}'")
            
            # Social media scraping
            social_scraper = SocialMediaScraper()
            social_queries = ["Ghana Stock Exchange", "GSE Ghana"]
            
            social_posts = social_scraper.scrape_all_platforms(social_queries, max_posts_per_platform=10)
            if social_posts:
                social_scraper.save_social_posts_to_db(social_posts, str(self.data_dir / "social_posts.db"))
                logger.info(f"[SUCCESS] Collected {len(social_posts)} social media posts")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error in initial data collection: {e}")
            return False
    
    def create_startup_scripts(self):
        """Create startup scripts for different components"""
        logger.info("Creating startup scripts...")
        
        # Main dashboard script
        dashboard_script = '''#!/bin/bash
echo "Starting GSE Sentiment Analysis Dashboard..."
cd "$(dirname "$0")"
streamlit run gse_sentiment_analysis_system.py --server.port 8501
'''
        
        # Manual sentiment input script
        manual_script = '''#!/bin/bash
echo "Starting Manual Sentiment Input Interface..."
cd "$(dirname "$0")"
streamlit run manual_sentiment_interface.py --server.port 8502
'''
        
        # Data collection script
        collection_script = '''#!/bin/bash
echo "Running data collection..."
cd "$(dirname "$0")"
python -c "
from gse_sentiment_analysis_system import GSESentimentAnalyzer
analyzer = GSESentimentAnalyzer()
analyzer.run_daily_collection()
print('Data collection completed!')
"
'''
        
        # Save scripts
        scripts = {
            'start_dashboard.sh': dashboard_script,
            'start_manual_input.sh': manual_script,
            'run_data_collection.sh': collection_script
        }
        
        for script_name, script_content in scripts.items():
            script_path = self.base_dir / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Make executable on Unix systems
            if os.name != 'nt':
                os.chmod(script_path, 0o755)
        
        # Windows batch files
        if os.name == 'nt':
            win_scripts = {
                'start_dashboard.bat': 'streamlit run gse_sentiment_analysis_system.py --server.port 8501',
                'start_manual_input.bat': 'streamlit run manual_sentiment_interface.py --server.port 8502',
                'run_data_collection.bat': 'python -c "from gse_sentiment_analysis_system import GSESentimentAnalyzer; analyzer = GSESentimentAnalyzer(); analyzer.run_daily_collection()"'
            }
            
            for script_name, command in win_scripts.items():
                with open(self.base_dir / script_name, 'w') as f:
                    f.write(f'@echo off\ncd /d "%~dp0"\n{command}\npause\n')
        
        logger.info("[SUCCESS] Startup scripts created")
        return True
    
    def run_complete_setup(self):
        """Run complete system setup"""
        logger.info("=" * 50)
        logger.info("GSE SENTIMENT ANALYSIS SYSTEM SETUP")
        logger.info("=" * 50)
        
        steps = [
            ("Installing requirements", self.install_requirements),
            ("Setting up databases", self.setup_databases),
            ("Loading GSE CSV data", self.load_gse_csv_data),
            ("Creating sample sentiment data", self.create_sample_sentiment_data),
            ("Creating configuration file", self.create_configuration_file),
            ("Creating startup scripts", self.create_startup_scripts),
            ("Running initial data collection", self.run_initial_data_collection)
        ]
        
        success_count = 0
        
        for step_name, step_function in steps:
            logger.info(f"\n[PROCESSING] {step_name}...")
            try:
                if step_function():
                    logger.info(f"[SUCCESS] {step_name} completed successfully")
                    success_count += 1
                else:
                    logger.error(f"[FAILED] {step_name} failed")
            except Exception as e:
                logger.error(f"[ERROR] {step_name} failed with error: {e}")
        
        logger.info("\n" + "=" * 50)
        logger.info(f"SETUP COMPLETED: {success_count}/{len(steps)} steps successful")
        logger.info("=" * 50)
        
        if success_count == len(steps):
            logger.info("\n🎉 GSE Sentiment Analysis System setup completed successfully!")
            self._print_next_steps()
        else:
            logger.warning(f"\n[WARNING]  Setup completed with {len(steps) - success_count} issues.")
            logger.warning("Please check the logs and resolve any errors before proceeding.")
        
        return success_count == len(steps)
    
    def _print_next_steps(self):
        """Print next steps for the user"""
        print("\n" + "=" * 60)
        print("🚀 NEXT STEPS:")
        print("=" * 60)
        print("1. Start the main dashboard:")
        if os.name == 'nt':
            print("   > start_dashboard.bat")
        else:
            print("   > ./start_dashboard.sh")
        print("   📍 Opens at: http://localhost:8501")
        
        print("\n2. Start manual sentiment input interface:")
        if os.name == 'nt':
            print("   > start_manual_input.bat")
        else:
            print("   > ./start_manual_input.sh")
        print("   📍 Opens at: http://localhost:8502")
        
        print("\n3. Run data collection manually:")
        if os.name == 'nt':
            print("   > run_data_collection.bat")
        else:
            print("   > ./run_data_collection.sh")
        
        print("\n4. Schedule automatic data collection:")
        print("   Set up a cron job (Linux/Mac) or Task Scheduler (Windows)")
        print("   to run data collection every 6 hours")
        
        print("\n📁 Important Files:")
        print(f"   • Configuration: config.json")
        print(f"   • Database: data/gse_sentiment.db")
        print(f"   • Logs: logs/gse_system.log")
        
        print("\n🔧 Customization:")
        print("   • Add your GSE CSV files to load real stock data")
        print("   • Modify news sources in news_scraper.py")
        print("   • Adjust sentiment models in gse_sentiment_analysis_system.py")
        
        print("=" * 60)

def main():
    """Main setup function"""
    print("GSE Sentiment Analysis System Setup")
    print("Developed for Ghana Stock Exchange Market Prediction")
    print("Based on your Chapter 3 Research Methodology\n")
    
    setup = GSESystemSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\n[SUCCESS] Setup completed successfully!")
    else:
        print("\n[ERROR] Setup completed with errors. Please check the logs.")
    
    return success

if __name__ == "__main__":
    main()