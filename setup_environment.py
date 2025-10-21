#!/usr/bin/env python3
"""
GSE SENTIMENT ANALYSIS - ENVIRONMENT SETUP
Installs all required dependencies and sets up the project environment.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error in {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def install_packages():
    """Install required Python packages"""
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "xgboost",
        "catboost",
        "lightgbm",
        "nltk",
        "textblob",
        "vaderSentiment",
        "transformers",
        "torch",
        "sqlalchemy",
        "streamlit",
        "requests",
        "beautifulsoup4",
        "schedule",
        "tqdm",
        "colorama"
    ]

    print("Installing Python packages...")
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"Warning: Failed to install {package}, continuing...")

def setup_nltk():
    """Setup NLTK data"""
    print("Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("NLTK setup completed")
    except Exception as e:
        print(f"Warning: NLTK setup failed: {e}")

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'logs', 'screenshots']

    print("Creating project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_sample_data():
    """Create sample database if it doesn't exist"""
    if not os.path.exists('gse_sentiment.db'):
        print("Creating sample sentiment database...")
        try:
            import sqlite3
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta

            # Create sample data
            np.random.seed(42)
            companies = ['GCB', 'ACCESS', 'MTNGH', 'EGH', 'CAL', 'GOIL', 'TOTAL', 'FML', 'UNIL']
            sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']

            # Generate sample records
            records = []
            base_date = datetime.now() - timedelta(days=365)

            for i in range(1000):
                company = np.random.choice(companies)
                sentiment_label = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])

                if sentiment_label == 'POSITIVE':
                    score = np.random.normal(0.4, 0.2)
                elif sentiment_label == 'NEGATIVE':
                    score = np.random.normal(-0.3, 0.2)
                else:
                    score = np.random.normal(0.05, 0.15)

                score = np.clip(score, -1, 1)

                record = {
                    'id': i + 1,
                    'company': company,
                    'sentiment_score': round(score, 3),
                    'sentiment_label': sentiment_label,
                    'source': np.random.choice(['news', 'twitter', 'facebook', 'expert']),
                    'timestamp': (base_date + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d %H:%M:%S'),
                    'text': f"Sample sentiment text for {company}"
                }
                records.append(record)

            # Create DataFrame and save to database
            df = pd.DataFrame(records)

            conn = sqlite3.connect('gse_sentiment.db')
            df.to_sql('sentiment_data', conn, if_exists='replace', index=False)
            conn.close()

            print("Sample database created with 1,000 records")
        except Exception as e:
            print(f"Warning: Failed to create sample database: {e}")
    else:
        print("Database already exists")

def test_installation():
    """Test that key packages are installed correctly"""
    print("Testing installation...")

    test_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('plotly', 'px'),
        ('sklearn', 'sklearn'),
        ('streamlit', 'st'),
        ('sqlite3', 'sqlite3')
    ]

    failed_imports = []

    for package, import_name in test_imports:
        try:
            __import__(import_name if import_name != package else package)
            print(f"[OK] {package} imported successfully")
        except ImportError:
            print(f"[FAIL] Failed to import {package}")
            failed_imports.append(package)

    if failed_imports:
        print(f"\\nWarning: Some packages failed to import: {', '.join(failed_imports)}")
        print("You may need to install them manually or check your Python environment")
    else:
        print("\\nAll key packages imported successfully!")

def main():
    """Main setup function"""
    print("GSE SENTIMENT ANALYSIS - ENVIRONMENT SETUP")
    print("=" * 50)

    # Check Python version
    print(f"Python version: {sys.version}")

    # Install packages
    install_packages()

    # Setup NLTK
    setup_nltk()

    # Create directories
    create_directories()

    # Create sample data
    create_sample_data()

    # Test installation
    test_installation()

    print("\\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("\\nNext steps:")
    print("1. Run the analysis notebook:")
    print("   jupyter notebook GSE_Sentiment_Analysis_Complete.ipynb")
    print("\\n2. Launch the dashboard:")
    print("   streamlit run simple_dashboard.py")
    print("\\n3. For data collection (optional):")
    print("   python news_scraper.py")
    print("\\nHappy analyzing!")

if __name__ == "__main__":
    main()