"""
System Testing Script for GSE Sentiment Analysis System
Comprehensive testing of all components
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """Comprehensive system testing"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.test_results = {}
        self.errors = []
    
    def test_imports(self):
        """Test if all required modules can be imported"""
        logger.info("Testing imports...")
        
        try:
            # Core system imports
            from gse_sentiment_analysis_system import GSESentimentAnalyzer
            from news_scraper import NewsScraper
            from social_media_scraper import SocialMediaScraper
            from gse_data_loader import GSEDataLoader
            from manual_sentiment_interface import ManualSentimentInterface
            
            # Required libraries
            import requests
            import beautifulsoup4
            import nltk
            import textblob
            import vaderSentiment
            import sklearn
            import matplotlib
            import seaborn
            import streamlit
            import plotly
            
            self.test_results['imports'] = True
            logger.info("✅ All imports successful")
            return True
            
        except ImportError as e:
            self.test_results['imports'] = False
            error_msg = f"Import error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_database_setup(self):
        """Test database creation and operations"""
        logger.info("Testing database setup...")
        
        try:
            # Test database creation
            test_db_path = self.base_dir / "test_database.db"
            conn = sqlite3.connect(test_db_path)
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value REAL
                )
            ''')
            
            # Insert test data
            cursor.execute('INSERT INTO test_table (name, value) VALUES (?, ?)', ('test', 1.0))
            
            # Query test data
            cursor.execute('SELECT * FROM test_table')
            result = cursor.fetchone()
            
            conn.commit()
            conn.close()
            
            # Clean up
            if test_db_path.exists():
                test_db_path.unlink()
            
            if result:
                self.test_results['database'] = True
                logger.info("✅ Database operations successful")
                return True
            else:
                raise Exception("No data retrieved from test table")
                
        except Exception as e:
            self.test_results['database'] = False
            error_msg = f"Database error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        logger.info("Testing sentiment analysis...")
        
        try:
            from gse_sentiment_analysis_system import GSESentimentAnalyzer
            
            analyzer = GSESentimentAnalyzer(db_path=":memory:")  # Use in-memory database
            
            # Test sentiment analysis
            test_texts = [
                "MTN Ghana reports excellent quarterly results with strong profit growth",
                "GCB Bank faces challenges amid economic downturn",
                "Ghana Stock Exchange trading remains stable today"
            ]
            
            for text in test_texts:
                score, label, confidence = analyzer.analyze_sentiment(text)
                
                # Validate results
                if not (-1 <= score <= 1):
                    raise Exception(f"Invalid sentiment score: {score}")
                if label not in ['positive', 'negative', 'neutral']:
                    raise Exception(f"Invalid sentiment label: {label}")
                if not (0 <= confidence <= 1):
                    raise Exception(f"Invalid confidence score: {confidence}")
            
            self.test_results['sentiment_analysis'] = True
            logger.info("✅ Sentiment analysis working correctly")
            return True
            
        except Exception as e:
            self.test_results['sentiment_analysis'] = False
            error_msg = f"Sentiment analysis error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_web_scraping(self):
        """Test web scraping functionality"""
        logger.info("Testing web scraping...")
        
        try:
            from news_scraper import NewsScraper
            
            scraper = NewsScraper(delay_range=(0.5, 1))  # Faster for testing
            
            # Test basic HTTP request
            test_url = "https://httpbin.org/get"
            response = scraper._make_request(test_url)
            
            if response and response.status_code == 200:
                self.test_results['web_scraping'] = True
                logger.info("✅ Web scraping functionality working")
                return True
            else:
                raise Exception("Failed to make basic HTTP request")
                
        except Exception as e:
            self.test_results['web_scraping'] = False
            error_msg = f"Web scraping error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            logger.warning("This might be due to internet connectivity issues")
            return False
    
    def test_data_processing(self):
        """Test data processing and feature extraction"""
        logger.info("Testing data processing...")
        
        try:
            from gse_data_loader import GSEDataLoader
            
            loader = GSEDataLoader()
            
            # Create sample data
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
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
            
            # Test technical indicators
            processed_data = loader._add_technical_indicators(sample_data)
            
            # Validate that indicators were added
            required_columns = ['MA_5', 'MA_10', 'MA_20', 'RSI', 'BB_Upper', 'BB_Lower']
            for col in required_columns:
                if col not in processed_data.columns:
                    raise Exception(f"Missing technical indicator: {col}")
            
            # Test feature extraction
            features = loader.get_stock_features(datetime.now())
            
            if not isinstance(features, dict) or len(features) == 0:
                raise Exception("Feature extraction failed")
            
            self.test_results['data_processing'] = True
            logger.info("✅ Data processing working correctly")
            return True
            
        except Exception as e:
            self.test_results['data_processing'] = False
            error_msg = f"Data processing error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_machine_learning(self):
        """Test machine learning model training"""
        logger.info("Testing machine learning models...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
            
            # Create sample training data
            np.random.seed(42)
            n_samples = 1000
            n_features = 10
            
            X = np.random.randn(n_samples, n_features)
            y = np.random.choice([0, 1], n_samples)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Test different models
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=10, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=100)
            }
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                if accuracy < 0.3:  # Very low threshold since data is random
                    logger.warning(f"Low accuracy for {model_name}: {accuracy:.3f}")
                
                logger.info(f"{model_name} accuracy: {accuracy:.3f}")
            
            self.test_results['machine_learning'] = True
            logger.info("✅ Machine learning models working correctly")
            return True
            
        except Exception as e:
            self.test_results['machine_learning'] = False
            error_msg = f"Machine learning error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_manual_input_system(self):
        """Test manual sentiment input functionality"""
        logger.info("Testing manual input system...")
        
        try:
            from manual_sentiment_interface import ManualSentimentInterface
            
            interface = ManualSentimentInterface(db_path=":memory:")
            
            # Test manual sentiment addition
            success = interface.save_manual_sentiment(
                company="MTN",
                news_type="earnings_report",
                content="Test news content for sentiment analysis",
                sentiment="positive",
                impact_level="medium",
                confidence="high",
                user_id="test_user"
            )
            
            if not success:
                raise Exception("Failed to save manual sentiment")
            
            # Test data retrieval
            summary = interface.get_sentiment_summary()
            
            if summary['total_entries'] != 1:
                raise Exception("Failed to retrieve saved sentiment data")
            
            self.test_results['manual_input'] = True
            logger.info("✅ Manual input system working correctly")
            return True
            
        except Exception as e:
            self.test_results['manual_input'] = False
            error_msg = f"Manual input error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_system_integration(self):
        """Test full system integration"""
        logger.info("Testing system integration...")
        
        try:
            from gse_sentiment_analysis_system import GSESentimentAnalyzer
            
            # Initialize with in-memory database
            analyzer = GSESentimentAnalyzer(db_path=":memory:")
            
            # Test sentiment feature extraction
            features = analyzer.get_sentiment_features("MTN")
            
            if not isinstance(features, dict):
                raise Exception("Failed to get sentiment features")
            
            # Test prediction (will use default features since no data)
            prediction = analyzer.predict_stock_movement("MTN")
            
            if 'prediction' not in prediction:
                raise Exception("Failed to generate prediction")
            
            # Test report generation
            report = analyzer.generate_report()
            
            if 'companies' not in report:
                raise Exception("Failed to generate report")
            
            self.test_results['system_integration'] = True
            logger.info("✅ System integration working correctly")
            return True
            
        except Exception as e:
            self.test_results['system_integration'] = False
            error_msg = f"System integration error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def test_configuration_files(self):
        """Test configuration and setup files"""
        logger.info("Testing configuration files...")
        
        try:
            # Check if essential files exist
            essential_files = [
                'requirements.txt',
                'gse_sentiment_analysis_system.py',
                'news_scraper.py',
                'social_media_scraper.py',
                'gse_data_loader.py',
                'manual_sentiment_interface.py',
                'setup_and_run.py'
            ]
            
            missing_files = []
            for file_name in essential_files:
                if not (self.base_dir / file_name).exists():
                    missing_files.append(file_name)
            
            if missing_files:
                raise Exception(f"Missing essential files: {missing_files}")
            
            # Test requirements.txt format
            requirements_file = self.base_dir / 'requirements.txt'
            with open(requirements_file, 'r') as f:
                requirements = f.read()
                if len(requirements.strip()) == 0:
                    raise Exception("Requirements file is empty")
            
            self.test_results['configuration'] = True
            logger.info("✅ Configuration files are valid")
            return True
            
        except Exception as e:
            self.test_results['configuration'] = False
            error_msg = f"Configuration error: {str(e)}"
            self.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE SYSTEM TESTING")
        logger.info("=" * 60)
        
        tests = [
            ("Import Tests", self.test_imports),
            ("Database Tests", self.test_database_setup),
            ("Sentiment Analysis Tests", self.test_sentiment_analysis),
            ("Web Scraping Tests", self.test_web_scraping),
            ("Data Processing Tests", self.test_data_processing),
            ("Machine Learning Tests", self.test_machine_learning),
            ("Manual Input Tests", self.test_manual_input_system),
            ("System Integration Tests", self.test_system_integration),
            ("Configuration Tests", self.test_configuration_files)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_function in tests:
            logger.info(f"\n🔄 Running {test_name}...")
            try:
                if test_function():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
                self.errors.append(f"{test_name}: {str(e)}")
        
        # Generate final report
        self._generate_test_report(passed_tests, total_tests)
        
        return passed_tests == total_tests
    
    def _generate_test_report(self, passed: int, total: int):
        """Generate and display test report"""
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        success_rate = (passed / total) * 100
        logger.info(f"Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate == 100:
            logger.info("🎉 ALL TESTS PASSED! System is ready for use.")
        elif success_rate >= 80:
            logger.info("✅ Most tests passed. System should work with minor issues.")
        elif success_rate >= 60:
            logger.warning("⚠️  Some critical tests failed. Review errors before proceeding.")
        else:
            logger.error("❌ Many tests failed. System needs significant fixes.")
        
        # Detailed results
        logger.info("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # Error summary
        if self.errors:
            logger.info("\nErrors Encountered:")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")
        
        # Recommendations
        logger.info("\nRecommendations:")
        if success_rate == 100:
            logger.info("  • System is ready for production use")
            logger.info("  • Run setup_and_run.py to initialize the system")
        else:
            logger.info("  • Review error messages above")
            logger.info("  • Install missing dependencies if needed")
            logger.info("  • Check internet connection for web scraping tests")
            logger.info("  • Ensure Python version is 3.8 or higher")
        
        logger.info("=" * 60)

def main():
    """Main testing function"""
    print("GSE Sentiment Analysis System - Comprehensive Testing")
    print("=" * 60)
    
    tester = SystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! System is ready for use.")
        print("Next step: Run 'python setup_and_run.py' to initialize the system.")
    else:
        print(f"\n⚠️  {len(tester.errors)} issues found. Please review the test results above.")
    
    return success

if __name__ == "__main__":
    main()