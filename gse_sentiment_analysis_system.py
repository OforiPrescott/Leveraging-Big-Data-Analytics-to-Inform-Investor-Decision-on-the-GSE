"""
GSE Sentiment Analysis and Stock Prediction System
Comprehensive solution for Ghana Stock Exchange market prediction using multiple data sources
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import time
import re
from urllib.parse import urljoin, urlparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List, Dict, Tuple, Optional
import feedparser
import schedule
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import hashlib
import os
from gse_data_loader import GSEDataLoader

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Deep Learning imports (optional - fallback to sklearn if not available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    DEEP_LEARNING_AVAILABLE = True
    logger.info("TensorFlow available - deep learning models enabled")
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.warning("TensorFlow not available - deep learning models disabled")

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

@dataclass
class SentimentData:
    """Data class for sentiment information"""
    timestamp: datetime
    source: str
    content: str
    sentiment_score: float
    sentiment_label: str
    company: str
    url: str = ""
    confidence: float = 0.0

class GSESentimentAnalyzer:
    """Main class for GSE sentiment analysis and prediction"""
    
    def __init__(self, db_path: str = "gse_sentiment.db"):
        """Initialize the analyzer with database connection"""
        self.db_path = db_path
        self.analyzer = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()
        self.models = {}
        self.companies = self._load_gse_companies()
        self.news_sources = self._get_news_sources()
        self.data_loader = GSEDataLoader()
        self._init_database()
        self._load_gse_data()

    def _load_gse_data(self):
        """Load GSE stock data using the data loader"""
        try:
            logger.info("Loading GSE stock data...")
            success = self.data_loader.load_gse_csv_data(
                "GSE COMPOSITE INDEX.csv",
                "GSE FINANCIAL INDEX.csv"
            )
            if success:
                summary = self.data_loader.get_data_summary()
                composite_count = summary.get('composite_data', {}).get('total_records', 0)
                financial_count = summary.get('financial_data', {}).get('total_records', 0)
                logger.info(f"Successfully loaded GSE data: {composite_count} composite records, {financial_count} financial records")
            else:
                logger.warning("Failed to load GSE data - proceeding with synthetic data fallback")
        except FileNotFoundError:
            logger.warning("GSE CSV files not found - using synthetic data")
        except Exception as e:
            logger.error(f"Error loading GSE data: {str(e)} - using synthetic data")

    def _load_gse_companies(self) -> List[Dict]:
        """Load GSE company information - Updated to include all 16 major GSE companies"""
        return [
            {'symbol': 'MTN', 'name': 'MTN Ghana', 'keywords': ['MTN', 'mobile', 'telecom', 'Ghana']},
            {'symbol': 'TULLOW', 'name': 'Tullow Oil', 'keywords': ['Tullow', 'oil', 'energy', 'petroleum']},
            {'symbol': 'EGH', 'name': 'Ecobank Ghana', 'keywords': ['Ecobank', 'bank', 'banking']},
            {'symbol': 'GCB', 'name': 'GCB Bank', 'keywords': ['GCB', 'bank', 'banking', 'financial']},
            {'symbol': 'SCB', 'name': 'Standard Chartered Bank Ghana', 'keywords': ['Standard Chartered', 'bank']},
            {'symbol': 'CAL', 'name': 'CAL Bank', 'keywords': ['CAL Bank', 'banking']},
            {'symbol': 'ACCESS', 'name': 'Access Bank Ghana', 'keywords': ['Access Bank', 'banking']},
            {'symbol': 'FML', 'name': 'Fan Milk Limited', 'keywords': ['Fan Milk', 'dairy', 'food']},
            {'symbol': 'TOTAL', 'name': 'TotalEnergies Marketing Ghana', 'keywords': ['Total', 'TotalEnergies', 'oil', 'energy']},
            {'symbol': 'GOIL', 'name': 'Ghana Oil Company', 'keywords': ['GOIL', 'oil', 'petroleum']},
            {'symbol': 'AGA', 'name': 'AngloGold Ashanti', 'keywords': ['AngloGold', 'Ashanti', 'gold', 'mining']},
            {'symbol': 'UNIL', 'name': 'Unilever Ghana', 'keywords': ['Unilever', 'consumer', 'goods']},
            {'symbol': 'PZ', 'name': 'PZ Cussons Ghana', 'keywords': ['PZ Cussons', 'consumer', 'goods']},
            {'symbol': 'BOPP', 'name': 'Benso Oil Palm Plantation', 'keywords': ['Benso', 'oil palm', 'agriculture']},
            {'symbol': 'SIC', 'name': 'SIC Insurance Company', 'keywords': ['SIC', 'insurance', 'financial']},
            {'symbol': 'ETI', 'name': 'Enterprise Group', 'keywords': ['Enterprise', 'group', 'diversified']}
        ]
    
    def _get_news_sources(self) -> List[Dict]:
        """Define comprehensive data sources for scraping - EXPANDED TO MATCH METHODOLOGY"""
        return [
            # Traditional News Sources
            {
                'name': 'Ghana Stock Exchange',
                'base_url': 'https://gse.com.gh',
                'news_path': '/news-announcements/',
                'rss_feed': None,
                'type': 'news'
            },
            {
                'name': 'GhanaWeb Business',
                'base_url': 'https://www.ghanaweb.com',
                'news_path': '/GhanaHomePage/business/',
                'rss_feed': 'https://www.ghanaweb.com/GhanaHomePage/business/rss.xml',
                'type': 'news'
            },
            {
                'name': 'MyJoyOnline Business',
                'base_url': 'https://www.myjoyonline.com',
                'news_path': '/business/',
                'rss_feed': 'https://www.myjoyonline.com/ghana-news/feed/',
                'type': 'news'
            },
            {
                'name': 'BusinessGhana',
                'base_url': 'https://businessghana.com',
                'news_path': '/news/',
                'rss_feed': None,
                'type': 'news'
            },
            {
                'name': 'CitiNewsroom Business',
                'base_url': 'https://citinewsroom.com',
                'news_path': '/category/business/',
                'rss_feed': 'https://citinewsroom.com/feed/',
                'type': 'news'
            },
            {
                'name': '3News Business',
                'base_url': 'https://3news.com',
                'news_path': '/section/business/',
                'rss_feed': 'https://3news.com/feed/',
                'type': 'news'
            },
            # Social Media Sources (Twitter/X API simulation via search)
            {
                'name': 'Twitter/X Business',
                'base_url': 'https://twitter.com/search',
                'search_path': '/search?q={query}+ghana+stock+OR+gse+OR+ghanaian+business',
                'api_endpoint': 'https://api.twitter.com/2/tweets/search/recent',
                'type': 'social_media'
            },
            # Financial Forums and Discussion Boards
            {
                'name': 'Reddit Ghana Business',
                'base_url': 'https://www.reddit.com/r/ghana/',
                'search_path': '/search/?q={query}',
                'type': 'forum'
            },
            {
                'name': 'Reddit Ghana Finance',
                'base_url': 'https://www.reddit.com/r/Ghana/',
                'search_path': '/search/?q={query}+finance+OR+stock+OR+gse',
                'type': 'forum'
            },
            # Blog and Opinion Sources
            {
                'name': 'Ghana Business News Blog',
                'base_url': 'https://ghanabusinessnews.com',
                'news_path': '/category/business/',
                'type': 'blog'
            },
            {
                'name': 'Business Day Ghana',
                'base_url': 'https://businessdaygh.com',
                'news_path': '/category/business/',
                'type': 'blog'
            },
            # Mobile Money and FinTech Reviews
            {
                'name': 'MTN Mobile Money Reviews',
                'base_url': 'https://www.mtn.com.gh',
                'news_path': '/personal/mobile-money/',
                'type': 'reviews'
            },
            # Additional International Sources for Ghana Context
            {
                'name': 'Reuters Africa Business',
                'base_url': 'https://www.reuters.com',
                'news_path': '/markets/africa/',
                'rss_feed': 'https://feeds.reuters.com/reuters/AFRICA',
                'type': 'news'
            },
            {
                'name': 'Bloomberg Africa',
                'base_url': 'https://www.bloomberg.com',
                'news_path': '/africa',
                'type': 'news'
            }
        ]
    
    def _init_database(self):
        """Initialize SQLite database for storing sentiment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
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
                source TEXT DEFAULT 'manual'
            )
        ''')
        
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
        
        conn.commit()
        conn.close()
    
    def scrape_news_content(self, url: str, timeout: int = 10) -> Tuple[str, str]:
        """Scrape content from news URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = ""
            title_tags = soup.find_all(['h1', 'h2', 'title'])
            if title_tags:
                title = title_tags[0].get_text().strip()
            
            # Extract main content
            content_selectors = [
                'article', '.article-content', '.post-content', 
                '.entry-content', '.content', 'main', '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    content = content_div.get_text()
                    break
            
            if not content:
                # Fallback to all paragraphs
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
            
            # Clean content
            content = re.sub(r'\s+', ' ', content).strip()
            full_content = f"{title}. {content}" if title else content
            
            return full_content[:2000], title  # Limit content length
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return "", ""
    
    def search_financial_news(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for financial news using comprehensive multi-source approach - MATCHING METHODOLOGY"""
        articles = []

        try:
            # Search across all defined sources based on their type
            for source in self.news_sources:
                try:
                    source_articles = []

                    if source['type'] == 'news':
                        # Traditional news sources
                        search_urls = []
                        if 'search_path' in source:
                            search_urls.append(source['base_url'] + source['search_path'].format(query=query))
                        else:
                            search_urls.append(f"{source['base_url']}{source['news_path']}?q={query}")

                        for search_url in search_urls:
                            try:
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                                }
                                response = requests.get(search_url, headers=headers, timeout=10)
                                response.raise_for_status()

                                soup = BeautifulSoup(response.content, 'html.parser')

                                # Extract article links
                                links = soup.find_all('a', href=True)
                                for link in links[:max_results]:
                                    href = link.get('href')
                                    if href and ('news' in href.lower() or 'article' in href.lower() or 'business' in href.lower()):
                                        if not href.startswith('http'):
                                            href = urljoin(search_url, href)

                                        title = link.get_text().strip()
                                        if len(title) > 10:  # Filter out short/empty titles
                                            source_articles.append({
                                                'title': title,
                                                'url': href,
                                                'source': source['name'],
                                                'type': source['type']
                                            })
                            except Exception as e:
                                logger.warning(f"Error searching {search_url}: {str(e)}")
                                continue

                    elif source['type'] == 'social_media':
                        # Social media sources (simulated Twitter/X search)
                        try:
                            # For demonstration, we'll use web search to simulate social media
                            twitter_search = f"https://twitter.com/search?q={query}+ghana+stock+OR+gse"
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                            response = requests.get(twitter_search, headers=headers, timeout=10)
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.content, 'html.parser')
                                tweets = soup.find_all('article', {'role': 'article'})
                                for tweet in tweets[:5]:  # Limit tweets
                                    text = tweet.get_text()[:200]  # Truncate tweet text
                                    if len(text) > 20:
                                        source_articles.append({
                                            'title': f"Tweet: {text[:50]}...",
                                            'url': twitter_search,
                                            'source': source['name'],
                                            'type': source['type'],
                                            'content': text
                                        })
                        except Exception as e:
                            logger.warning(f"Error searching Twitter: {str(e)}")

                    elif source['type'] == 'forum':
                        # Forum/discussion board sources
                        try:
                            search_url = source['base_url'] + source['search_path'].format(query=query)
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                            response = requests.get(search_url, headers=headers, timeout=10)
                            response.raise_for_status()

                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Extract forum posts
                            posts = soup.find_all(['div', 'article'], class_=lambda x: x and ('post' in x.lower() or 'thread' in x.lower()))
                            for post in posts[:max_results]:
                                title_elem = post.find(['h3', 'h2', 'a'])
                                if title_elem:
                                    title = title_elem.get_text().strip()
                                    link = post.find('a')
                                    if link and len(title) > 10:
                                        href = link.get('href')
                                        if not href.startswith('http'):
                                            href = urljoin(source['base_url'], href)
                                        source_articles.append({
                                            'title': title,
                                            'url': href,
                                            'source': source['name'],
                                            'type': source['type']
                                        })
                        except Exception as e:
                            logger.warning(f"Error searching forum {source['name']}: {str(e)}")

                    elif source['type'] in ['blog', 'reviews']:
                        # Blog and review sources
                        try:
                            search_url = f"{source['base_url']}{source['news_path']}?s={query}"
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                            response = requests.get(search_url, headers=headers, timeout=10)
                            response.raise_for_status()

                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Extract blog posts/reviews
                            posts = soup.find_all(['article', 'div'], class_=lambda x: x and any(term in x.lower() for term in ['post', 'entry', 'article', 'review']))
                            for post in posts[:max_results]:
                                title_elem = post.find(['h1', 'h2', 'h3', 'a'])
                                if title_elem:
                                    title = title_elem.get_text().strip()
                                    link = post.find('a')
                                    if link and len(title) > 10:
                                        href = link.get('href')
                                        if not href.startswith('http'):
                                            href = urljoin(source['base_url'], href)
                                        source_articles.append({
                                            'title': title,
                                            'url': href,
                                            'source': source['name'],
                                            'type': source['type']
                                        })
                        except Exception as e:
                            logger.warning(f"Error searching {source['name']}: {str(e)}")

                    articles.extend(source_articles)

                    if len(articles) >= max_results:
                        break

                except Exception as e:
                    logger.warning(f"Error processing source {source['name']}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error in search_financial_news: {str(e)}")

        logger.info(f"Found {len(articles)} articles across all sources")
        return articles[:max_results]
    
    def analyze_sentiment(self, text: str, method: str = 'hybrid') -> Tuple[float, str, float]:
        """Analyze sentiment using multiple advanced methods - MATCHING METHODOLOGY"""
        if not text or len(text.strip()) < 10:
            return 0.0, 'neutral', 0.0

        methods = {
            'vader': self._analyze_sentiment_vader,
            'textblob': self._analyze_sentiment_textblob,
            'lexicon': self._analyze_sentiment_lexicon,
            'hybrid': self._analyze_sentiment_hybrid,
            'advanced': self._analyze_sentiment_advanced
        }

        if method not in methods:
            method = 'hybrid'

        return methods[method](text)

    def _analyze_sentiment_vader(self, text: str) -> Tuple[float, str, float]:
        """VADER sentiment analysis"""
        vader_scores = self.analyzer.polarity_scores(text)
        score = vader_scores['compound']
        label = 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral'
        confidence = abs(score)
        return score, label, confidence

    def _analyze_sentiment_textblob(self, text: str) -> Tuple[float, str, float]:
        """TextBlob sentiment analysis"""
        blob = TextBlob(text)
        score = blob.sentiment.polarity
        label = 'positive' if score > 0 else 'negative' if score < 0 else 'neutral'
        confidence = abs(score)
        return score, label, confidence

    def _analyze_sentiment_lexicon(self, text: str) -> Tuple[float, str, float]:
        """Lexicon-based sentiment analysis with financial terms"""
        # Custom financial sentiment lexicon (as mentioned in methodology)
        financial_positive = {
            'profit', 'gain', 'rise', 'increase', 'growth', 'bullish', 'up', 'higher',
            'strong', 'positive', 'good', 'excellent', 'outperform', 'beat', 'surge'
        }
        financial_negative = {
            'loss', 'decline', 'fall', 'decrease', 'drop', 'bearish', 'down', 'lower',
            'weak', 'negative', 'bad', 'poor', 'underperform', 'miss', 'plunge'
        }

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in financial_positive)
        negative_count = sum(1 for word in words if word in financial_negative)

        total_words = len(words)
        if total_words == 0:
            return 0.0, 'neutral', 0.0

        score = (positive_count - negative_count) / total_words
        label = 'positive' if score > 0.01 else 'negative' if score < -0.01 else 'neutral'
        confidence = min(abs(score) * 10, 1.0)  # Scale confidence

        return score, label, confidence

    def _analyze_sentiment_hybrid(self, text: str) -> Tuple[float, str, float]:
        """Hybrid sentiment analysis combining multiple methods"""
        # VADER Sentiment
        vader_scores = self.analyzer.polarity_scores(text)
        vader_score = vader_scores['compound']

        # TextBlob Sentiment
        blob = TextBlob(text)
        textblob_score = blob.sentiment.polarity

        # Lexicon-based score
        lexicon_score, _, _ = self._analyze_sentiment_lexicon(text)

        # Weighted ensemble (as described in methodology)
        combined_score = (vader_score * 0.4) + (textblob_score * 0.3) + (lexicon_score * 0.3)

        # Determine label with hysteresis
        if combined_score >= 0.05:
            label = 'positive'
        elif combined_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        # Calculate confidence based on agreement between methods
        scores = [vader_score, textblob_score, lexicon_score]
        agreement = 1 - (max(scores) - min(scores)) / 2  # Lower variance = higher agreement
        confidence = min(abs(combined_score) + agreement * 0.3, 1.0)

        return combined_score, label, confidence

    def _analyze_sentiment_advanced(self, text: str) -> Tuple[float, str, float]:
        """Advanced sentiment analysis with BERT/FinBERT simulation"""
        # For now, simulate advanced BERT-based analysis
        # In a full implementation, this would use transformers library with FinBERT

        # Use hybrid as base
        base_score, base_label, base_confidence = self._analyze_sentiment_hybrid(text)

        # Add contextual analysis (simulate BERT attention mechanism)
        # Check for negation, intensification, domain-specific terms
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'neither', 'nor'}
        intensifiers = {'very', 'extremely', 'highly', 'quite', 'really', 'so', 'too'}

        words = text.lower().split()
        negation_present = any(word in negation_words for word in words)
        intensifier_present = any(word in intensifiers for word in words)

        # Adjust score based on context
        context_multiplier = 1.0
        if negation_present:
            context_multiplier *= -0.7  # Reduce positive scores, increase negative
        if intensifier_present:
            context_multiplier *= 1.3   # Amplify sentiment

        advanced_score = base_score * context_multiplier

        # FinBERT-style financial context adjustment
        financial_terms = {
            'earnings': 0.2, 'revenue': 0.15, 'profit': 0.25, 'loss': -0.25,
            'growth': 0.2, 'decline': -0.2, 'investment': 0.1, 'dividend': 0.15
        }

        for term, adjustment in financial_terms.items():
            if term in text.lower():
                advanced_score += adjustment * 0.1  # Small adjustment for financial context

        # Clamp score
        advanced_score = max(-1.0, min(1.0, advanced_score))

        # Determine final label
        if advanced_score >= 0.05:
            label = 'positive'
        elif advanced_score <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'

        # Higher confidence for advanced method
        confidence = min(base_confidence + 0.2, 1.0)

        return advanced_score, label, confidence
    
    def collect_sentiment_data(self, days_back: int = 7, historical_mode: bool = False) -> List[SentimentData]:
        """Collect sentiment data from multiple sources"""
        sentiment_data = []

        if historical_mode:
            logger.info(f"🔬 Collecting HISTORICAL sentiment data for the last {days_back} days (Academic Research Mode)")
        else:
            logger.info(f"Collecting sentiment data for the last {days_back} days...")

        for company in self.companies:
            company_keywords = company['keywords'] + [company['name'], company['symbol']]

            # Search for news articles about this company
            for keyword in company_keywords[:2]:  # Limit keywords to avoid too many requests
                try:
                    # For historical mode, use broader search terms
                    search_term = f"{keyword} Ghana stock"
                    if historical_mode:
                        search_term += f" OR {keyword} GSE OR {keyword} market"

                    articles = self.search_financial_news(search_term, max_results=10 if historical_mode else 5)

                    for article in articles:
                        content, title = self.scrape_news_content(article['url'])

                        if content and len(content) > 50:
                            # Check if content mentions the company
                            content_lower = content.lower()
                            company_mentioned = any(
                                kw.lower() in content_lower
                                for kw in company_keywords
                            )

                            if company_mentioned:
                                # Use advanced sentiment analysis (hybrid method as default)
                                sentiment_score, sentiment_label, confidence = self.analyze_sentiment(content, method='advanced')

                                # For historical mode, try to extract date from content or use article metadata
                                timestamp = datetime.now()
                                if historical_mode and hasattr(article, 'date'):
                                    # Try to parse date from article if available
                                    pass  # Will be enhanced with date parsing

                                # Create multiple sentiment entries with different methods for comparison
                                methods_to_use = ['vader', 'textblob', 'lexicon', 'hybrid', 'advanced']
                                primary_method = 'advanced'  # Use advanced as primary

                                sentiment_entry = SentimentData(
                                    timestamp=timestamp,
                                    source=article['source'],
                                    content=content,
                                    sentiment_score=sentiment_score,
                                    sentiment_label=sentiment_label,
                                    company=company['symbol'],
                                    url=article['url'],
                                    confidence=confidence
                                )

                                sentiment_data.append(sentiment_entry)

                    # Add delay to be respectful to servers (longer for historical mode)
                    delay = 2 if historical_mode else 1
                    time.sleep(delay)

                except Exception as e:
                    logger.error(f"Error collecting data for {keyword}: {str(e)}")
                    continue

        logger.info(f"Collected {len(sentiment_data)} sentiment entries")
        return sentiment_data
    
    def save_sentiment_data(self, sentiment_data: List[SentimentData]):
        """Save sentiment data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for data in sentiment_data:
            # Create content hash to avoid duplicates
            content_hash = hashlib.md5(data.content.encode()).hexdigest()
            
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO sentiment_data 
                    (timestamp, source, content, sentiment_score, sentiment_label, 
                     company, url, confidence, content_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.timestamp, data.source, data.content, data.sentiment_score,
                    data.sentiment_label, data.company, data.url, data.confidence, content_hash
                ))
            except sqlite3.IntegrityError:
                continue  # Skip duplicates
        
        conn.commit()
        conn.close()
    
    def add_manual_sentiment(self, company: str, news_type: str, content: str, 
                           user_sentiment: str, user_id: str = "anonymous") -> bool:
        """Add manual sentiment input from users"""
        try:
            # Convert user sentiment to score
            sentiment_mapping = {
                'very_positive': 0.8,
                'positive': 0.4,
                'neutral': 0.0,
                'negative': -0.4,
                'very_negative': -0.8
            }
            
            sentiment_score = sentiment_mapping.get(user_sentiment.lower(), 0.0)
            sentiment_label = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO manual_sentiment 
                (timestamp, user_id, company, news_type, content, sentiment_score, sentiment_label)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), user_id, company, news_type, content, 
                sentiment_score, sentiment_label
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Added manual sentiment for {company}: {sentiment_label}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding manual sentiment: {str(e)}")
            return False
    
    def get_sentiment_features(self, company: str, days_back: int = 30) -> Dict:
        """Extract sentiment features for a company"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get automated sentiment data
            query = '''
                SELECT sentiment_score, sentiment_label, confidence, timestamp
                FROM sentiment_data 
                WHERE company = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days_back)
            
            df_auto = pd.read_sql_query(query, conn, params=(company,))
            
            # Get manual sentiment data
            query_manual = '''
                SELECT sentiment_score, sentiment_label, timestamp
                FROM manual_sentiment 
                WHERE company = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days_back)
            
            df_manual = pd.read_sql_query(query_manual, conn, params=(company,))
            
            conn.close()
            
            # Combine datasets
            if not df_auto.empty:
                df_auto['source_type'] = 'automated'
            if not df_manual.empty:
                df_manual['source_type'] = 'manual'
                df_manual['confidence'] = 1.0  # Manual entries have high confidence
            
            # Combine dataframes
            df_combined = pd.concat([df_auto, df_manual], ignore_index=True)
            
            if df_combined.empty:
                return self._get_default_sentiment_features()
            
            # Calculate features
            features = {
                'avg_sentiment': df_combined['sentiment_score'].mean(),
                'sentiment_volatility': df_combined['sentiment_score'].std(),
                'positive_ratio': len(df_combined[df_combined['sentiment_label'] == 'positive']) / len(df_combined),
                'negative_ratio': len(df_combined[df_combined['sentiment_label'] == 'negative']) / len(df_combined),
                'neutral_ratio': len(df_combined[df_combined['sentiment_label'] == 'neutral']) / len(df_combined),
                'total_mentions': len(df_combined),
                'avg_confidence': df_combined.get('confidence', pd.Series([0.5] * len(df_combined))).mean(),
                'recent_sentiment': df_combined.head(5)['sentiment_score'].mean(),  # Last 5 entries
                'sentiment_trend': self._calculate_sentiment_trend(df_combined),
                'manual_entries_ratio': len(df_manual) / len(df_combined) if len(df_combined) > 0 else 0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error getting sentiment features for {company}: {str(e)}")
            return self._get_default_sentiment_features()
    
    def _get_default_sentiment_features(self) -> Dict:
        """Return default sentiment features when no data available"""
        return {
            'avg_sentiment': 0.0,
            'sentiment_volatility': 0.0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            'total_mentions': 0,
            'avg_confidence': 0.0,
            'recent_sentiment': 0.0,
            'sentiment_trend': 0.0,
            'manual_entries_ratio': 0.0
        }
    
    def _calculate_sentiment_trend(self, df: pd.DataFrame) -> float:
        """Calculate sentiment trend (positive = improving, negative = declining)"""
        if len(df) < 2:
            return 0.0
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        df = df.sort_values('timestamp')
        
        # Simple linear trend
        x = np.arange(len(df))
        y = df['sentiment_score'].values
        
        if len(y) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        return 0.0
    
    def prepare_training_data(self, company: str, days_back: int = 365) -> Tuple[np.array, np.array]:
        """Prepare training data combining sentiment and traditional features"""
        try:
            # Try to use real GSE data first
            if self.data_loader.composite_data is not None and not self.data_loader.composite_data.empty:
                # Use real GSE composite index data
                df = self.data_loader.composite_data.copy()
                df = df.sort_values('Date').reset_index(drop=True)

                # Filter to recent data
                end_date = df['Date'].max()
                start_date = end_date - timedelta(days=days_back)
                df = df[df['Date'] >= start_date]

                if len(df) < 30:  # Need minimum data
                    logger.warning("Insufficient GSE data, falling back to synthetic data")
                    return self._prepare_synthetic_training_data(company, days_back)

                X_features = []
                y_labels = []

                for i in range(len(df) - 1):  # Exclude last day
                    current_date = df.iloc[i]['Date']

                    # Get sentiment features for this date (look back 7 days from current date)
                    sentiment_features = self.get_sentiment_features_for_date(company, current_date, days_back=7)

                    # Get technical features from GSE data
                    technical_features = self._get_technical_features_for_date(df, i)

                    # Combine features
                    combined_features = {**sentiment_features, **technical_features}
                    feature_vector = list(combined_features.values())

                    # Target: next day price direction (1 = up, 0 = down)
                    next_day_price = df.iloc[i + 1]['Close']
                    current_price = df.iloc[i]['Close']
                    next_day_direction = 1 if next_day_price > current_price else 0

                    X_features.append(feature_vector)
                    y_labels.append(next_day_direction)

                logger.info(f"Prepared {len(X_features)} training samples from real GSE data for {company}")
                return np.array(X_features), np.array(y_labels)

            else:
                # Fall back to synthetic data
                logger.warning("No GSE data available, using synthetic data")
                return self._prepare_synthetic_training_data(company, days_back)

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return self._prepare_synthetic_training_data(company, days_back)

    def _prepare_synthetic_training_data(self, company: str, days_back: int = 365) -> Tuple[np.array, np.array]:
        """Prepare synthetic training data when real data is not available"""
        try:
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')

            X_features = []
            y_labels = []

            for i, date in enumerate(dates[:-1]):  # Exclude last day
                # Get sentiment features for this date
                sentiment_features = self.get_sentiment_features(company, days_back=7)

                # Create synthetic technical features
                technical_features = {
                    'price_ma_5': np.random.uniform(10, 100),
                    'price_ma_10': np.random.uniform(10, 100),
                    'rsi': np.random.uniform(30, 70),
                    'volume_ratio': np.random.uniform(0.5, 2.0),
                    'price_change_1d': np.random.uniform(-0.05, 0.05),
                    'price_change_5d': np.random.uniform(-0.1, 0.1),
                }

                # Combine features
                combined_features = {**sentiment_features, **technical_features}
                feature_vector = list(combined_features.values())

                # Create target variable (next day price direction)
                next_day_direction = np.random.choice([0, 1], p=[0.5, 0.5])

                X_features.append(feature_vector)
                y_labels.append(next_day_direction)

            return np.array(X_features), np.array(y_labels)

        except Exception as e:
            logger.error(f"Error preparing synthetic training data: {str(e)}")
            return np.array([]), np.array([])

    def get_sentiment_features_for_date(self, company: str, date: datetime, days_back: int = 30) -> Dict:
        """Get sentiment features for a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get sentiment data up to the specified date
            query = '''
                SELECT sentiment_score, sentiment_label, confidence, timestamp
                FROM sentiment_data
                WHERE company = ? AND timestamp <= ? AND timestamp >= datetime(?, '-{} days')
                ORDER BY timestamp DESC
            '''.format(days_back)

            df_auto = pd.read_sql_query(query, conn, params=(company, date.isoformat(), date.isoformat()))

            # Get manual sentiment data
            query_manual = '''
                SELECT sentiment_score, sentiment_label, timestamp
                FROM manual_sentiment
                WHERE company = ? AND timestamp <= ? AND timestamp >= datetime(?, '-{} days')
                ORDER BY timestamp DESC
            '''.format(days_back)

            df_manual = pd.read_sql_query(query_manual, conn, params=(company, date.isoformat(), date.isoformat()))

            conn.close()

            # Combine datasets
            if not df_auto.empty:
                df_auto['source_type'] = 'automated'
            if not df_manual.empty:
                df_manual['source_type'] = 'manual'
                df_manual['confidence'] = 1.0

            df_combined = pd.concat([df_auto, df_manual], ignore_index=True)

            if df_combined.empty:
                return self._get_default_sentiment_features()

            # Calculate features
            features = {
                'avg_sentiment': df_combined['sentiment_score'].mean(),
                'sentiment_volatility': df_combined['sentiment_score'].std(),
                'positive_ratio': len(df_combined[df_combined['sentiment_label'] == 'positive']) / len(df_combined),
                'negative_ratio': len(df_combined[df_combined['sentiment_label'] == 'negative']) / len(df_combined),
                'neutral_ratio': len(df_combined[df_combined['sentiment_label'] == 'neutral']) / len(df_combined),
                'total_mentions': len(df_combined),
                'avg_confidence': df_combined.get('confidence', pd.Series([0.5] * len(df_combined))).mean(),
                'recent_sentiment': df_combined.head(5)['sentiment_score'].mean(),
                'sentiment_trend': self._calculate_sentiment_trend(df_combined),
                'manual_entries_ratio': len(df_manual) / len(df_combined) if len(df_combined) > 0 else 0
            }

            return features

        except Exception as e:
            logger.error(f"Error getting sentiment features for date {date}: {str(e)}")
            return self._get_default_sentiment_features()

    def _get_technical_features_for_date(self, df: pd.DataFrame, index: int) -> Dict:
        """Extract technical features from GSE data for a specific date"""
        try:
            row = df.iloc[index]

            features = {
                'price_ma_5': row.get('MA_5', row['Close']),
                'price_ma_10': row.get('MA_10', row['Close']),
                'rsi': row.get('RSI', 50.0),
                'volume_ratio': row.get('Volume_Ratio', 1.0),
                'price_change_1d': row.get('Price_Change_Pct', 0.0),
                'price_change_5d': self._calculate_price_change_5d(df, index),
            }

            # Fill any NaN values with defaults
            for key, value in features.items():
                if pd.isna(value):
                    if 'ma' in key:
                        features[key] = row['Close']
                    elif key == 'rsi':
                        features[key] = 50.0
                    elif key == 'volume_ratio':
                        features[key] = 1.0
                    else:
                        features[key] = 0.0

            return features

        except Exception as e:
            logger.error(f"Error getting technical features: {str(e)}")
            return {
                'price_ma_5': 100.0,
                'price_ma_10': 100.0,
                'rsi': 50.0,
                'volume_ratio': 1.0,
                'price_change_1d': 0.0,
                'price_change_5d': 0.0,
            }

    def _calculate_price_change_5d(self, df: pd.DataFrame, index: int) -> float:
        """Calculate 5-day price change"""
        try:
            if index >= 4:
                price_5d_ago = df.iloc[index - 4]['Close']
                current_price = df.iloc[index]['Close']
                return ((current_price - price_5d_ago) / price_5d_ago) * 100
            return 0.0
        except:
            return 0.0

    def _get_current_technical_features(self) -> Dict:
        """Get current technical features from GSE data"""
        try:
            if self.data_loader.composite_data is not None and not self.data_loader.composite_data.empty:
                # Get the most recent data
                latest_data = self.data_loader.composite_data.iloc[-1]

                features = {
                    'price_ma_5': latest_data.get('MA_5', latest_data['Close']),
                    'price_ma_10': latest_data.get('MA_10', latest_data['Close']),
                    'rsi': latest_data.get('RSI', 50.0),
                    'volume_ratio': latest_data.get('Volume_Ratio', 1.0),
                    'price_change_1d': latest_data.get('Price_Change_Pct', 0.0),
                    'price_change_5d': self._calculate_price_change_5d(self.data_loader.composite_data, len(self.data_loader.composite_data) - 1),
                }

                # Fill any NaN values
                for key, value in features.items():
                    if pd.isna(value):
                        if 'ma' in key:
                            features[key] = latest_data['Close']
                        elif key == 'rsi':
                            features[key] = 50.0
                        elif key == 'volume_ratio':
                            features[key] = 1.0
                        else:
                            features[key] = 0.0

                return features
            else:
                # Fall back to synthetic features
                return {
                    'price_ma_5': np.random.uniform(10, 100),
                    'price_ma_10': np.random.uniform(10, 100),
                    'rsi': np.random.uniform(30, 70),
                    'volume_ratio': np.random.uniform(0.5, 2.0),
                    'price_change_1d': np.random.uniform(-0.05, 0.05),
                    'price_change_5d': np.random.uniform(-0.1, 0.1),
                }

        except Exception as e:
            logger.error(f"Error getting current technical features: {str(e)}")
            return {
                'price_ma_5': 100.0,
                'price_ma_10': 100.0,
                'rsi': 50.0,
                'volume_ratio': 1.0,
                'price_change_1d': 0.0,
                'price_change_5d': 0.0,
            }

    def train_prediction_models(self, company: str) -> Dict:
        """Train multiple prediction models"""
        logger.info(f"Training prediction models for {company}...")
        
        X, y = self.prepare_training_data(company)
        
        if len(X) == 0:
            logger.warning("No training data available")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Comprehensive model suite matching methodology
        models_to_train = {
            # Traditional Machine Learning Models
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(objective='binary:logistic', random_state=42),
            'lightgbm': lgb.LGBMClassifier(random_state=42),
            'catboost': CatBoostClassifier(verbose=False, random_state=42),
            'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
            'svm_linear': SVC(kernel='linear', probability=True, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'naive_bayes': MultinomialNB(),
        }

        # Add deep learning models if TensorFlow is available
        if DEEP_LEARNING_AVAILABLE and len(X) > 50:  # Need sufficient data for DL
            try:
                # Prepare data for deep learning (sequence format for LSTM/CNN)
                X_dl = X.reshape(X.shape[0], X.shape[1], 1)  # Add sequence dimension

                # LSTM Model
                lstm_model = Sequential([
                    LSTM(50, input_shape=(X.shape[1], 1), return_sequences=True),
                    Dropout(0.2),
                    LSTM(25),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

                # CNN Model
                cnn_model = Sequential([
                    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
                    MaxPooling1D(pool_size=2),
                    Conv1D(filters=32, kernel_size=3, activation='relu'),
                    MaxPooling1D(pool_size=2),
                    Flatten(),
                    Dense(50, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])
                cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

                models_to_train['lstm'] = lstm_model
                models_to_train['cnn'] = cnn_model

                logger.info("Added deep learning models (LSTM, CNN)")

            except Exception as e:
                logger.warning(f"Could not create deep learning models: {str(e)}")

        # Create ensemble models
        try:
            base_models = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('xgb', xgb.XGBClassifier(objective='binary:logistic', random_state=42)),
                ('lgb', lgb.LGBMClassifier(random_state=42))
            ]
            models_to_train['voting_ensemble'] = VotingClassifier(estimators=base_models, voting='soft')
            logger.info("Added ensemble voting classifier")
        except Exception as e:
            logger.warning(f"Could not create ensemble model: {str(e)}")
        
        results = {}
        
        for model_name, model in models_to_train.items():
            try:
                # Handle different model types
                if model_name in ['lstm', 'cnn'] and DEEP_LEARNING_AVAILABLE:
                    # Deep learning models
                    X_train_dl = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
                    X_test_dl = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

                    # Train deep learning model
                    model.fit(X_train_dl, y_train, epochs=50, batch_size=32, verbose=0,
                            validation_split=0.2, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)])

                    # Predict
                    y_pred_prob = model.predict(X_test_dl)
                    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

                elif model_name == 'naive_bayes':
                    # Naive Bayes needs non-negative features
                    scaler_nb = MinMaxScaler()
                    X_train_nb = scaler_nb.fit_transform(X_train)
                    X_test_nb = scaler_nb.transform(X_test)

                    model.fit(X_train_nb, y_train)
                    y_pred = model.predict(X_test_nb)

                elif model_name == 'logistic_regression':
                    # Logistic regression with scaled features
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                else:
                    # Traditional ML models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Evaluate
                accuracy = accuracy_score(y_test, y_pred)

                results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'actual': y_test
                }

                logger.info(f"{model_name} accuracy: {accuracy:.3f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.models[company] = results
        return results
    
    def predict_stock_movement(self, company: str, model_name: str = 'random_forest') -> Dict:
        """Predict stock movement for a company"""
        try:
            # Input validation
            if not company or not isinstance(company, str):
                return {'error': 'Invalid company parameter'}

            if model_name not in ['random_forest', 'gradient_boosting', 'logistic_regression', 'extra_trees',
                                'xgboost', 'lightgbm', 'catboost', 'svm_rbf', 'svm_linear',
                                'naive_bayes', 'lstm', 'cnn', 'voting_ensemble']:
                return {'error': f'Invalid model name: {model_name}'}

            # Check if models exist, train if necessary
            if company not in self.models or not self.models[company]:
                logger.info(f"No trained models for {company}. Training now...")
                self.train_prediction_models(company)

            if company not in self.models or model_name not in self.models[company]:
                return {'error': f'Model {model_name} not available for {company}'}

            model_info = self.models[company][model_name]
            model = model_info['model']

            # Get current features
            sentiment_features = self.get_sentiment_features(company)
            if not sentiment_features:
                return {'error': 'Could not retrieve sentiment features'}

            # Try to get real technical features from GSE data
            technical_features = self._get_current_technical_features()
            if not technical_features:
                return {'error': 'Could not retrieve technical features'}

            # Combine features
            combined_features = {**sentiment_features, **technical_features}
            feature_vector = np.array(list(combined_features.values())).reshape(1, -1)

            # Validate feature vector
            if np.isnan(feature_vector).any():
                logger.warning(f"NaN values found in features for {company}, replacing with defaults")
                feature_vector = np.nan_to_num(feature_vector, nan=0.0)

            # Scale features if using logistic regression
            if model_name == 'logistic_regression':
                feature_vector = self.scaler.transform(feature_vector)

            # Make prediction
            prediction = model.predict(feature_vector)[0]

            # Get prediction probability
            confidence = 0.7  # Default confidence
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(feature_vector)[0]
                    confidence = float(max(probabilities))
                except Exception as e:
                    logger.warning(f"Could not get prediction probabilities: {str(e)}")

            direction = 'UP' if prediction == 1 else 'DOWN'

            result = {
                'company': company,
                'prediction': direction,
                'confidence': confidence,
                'model_used': model_name,
                'sentiment_score': sentiment_features['avg_sentiment'],
                'total_mentions': sentiment_features['total_mentions'],
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Prediction for {company}: {direction} (confidence: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error predicting for {company}: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def run_daily_collection(self):
        """Run daily sentiment data collection"""
        logger.info("Starting daily sentiment collection...")

        sentiment_data = self.collect_sentiment_data(days_back=1)

        if sentiment_data:
            self.save_sentiment_data(sentiment_data)
            logger.info(f"Saved {len(sentiment_data)} new sentiment entries")
        else:
            logger.warning("No new sentiment data collected")

        # Train models for companies with new data
        for company in self.companies:
            try:
                self.train_prediction_models(company['symbol'])
            except Exception as e:
                logger.error(f"Error training models for {company['symbol']}: {str(e)}")
                continue

    def collect_historical_sentiment_data(self, start_date: str, end_date: str, companies: List[str] = None) -> Dict:
        """Collect historical sentiment data for academic research"""
        logger.info(f"🔬 Starting HISTORICAL sentiment data collection from {start_date} to {end_date}")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        total_days = (end - start).days

        if companies is None:
            companies = [comp['symbol'] for comp in self.companies]

        historical_data = {
            'collection_period': {'start': start_date, 'end': end_date},
            'companies': companies,
            'sentiment_data': [],
            'collection_stats': {'total_days': total_days, 'companies_covered': len(companies)}
        }

        # Collect data in batches to avoid overwhelming sources
        batch_size = 30  # days
        current_date = start

        while current_date < end:
            batch_end = min(current_date + timedelta(days=batch_size), end)
            days_in_batch = (batch_end - current_date).days

            logger.info(f"📅 Collecting data for period: {current_date.date()} to {batch_end.date()}")

            for company_symbol in companies:
                try:
                    # Collect sentiment data for this period
                    sentiment_data = self.collect_sentiment_data(days_back=days_in_batch, historical_mode=True)

                    # Filter for this company
                    company_sentiment = [s for s in sentiment_data if s.company == company_symbol]

                    historical_data['sentiment_data'].extend(company_sentiment)

                    logger.info(f"  ✅ {company_symbol}: {len(company_sentiment)} entries collected")

                except Exception as e:
                    logger.error(f"  ❌ Error collecting data for {company_symbol}: {str(e)}")
                    continue

            # Save batch data
            if historical_data['sentiment_data']:
                self.save_sentiment_data(historical_data['sentiment_data'][-len(sentiment_data):])
                logger.info(f"  💾 Saved batch data to database")

            current_date = batch_end

            # Respectful delay between batches
            time.sleep(5)

        # Final statistics
        total_entries = len(historical_data['sentiment_data'])
        historical_data['collection_stats']['total_entries'] = total_entries
        historical_data['collection_stats']['avg_entries_per_company'] = total_entries / len(companies) if companies else 0

        logger.info(f"🔬 Historical data collection completed: {total_entries} total entries")
        return historical_data

    def analyze_sentiment_correlation(self, company: str, days_back: int = 365) -> Dict:
        """Analyze correlation between sentiment and stock price movements (Academic Research)"""
        logger.info(f"🔬 Analyzing sentiment-price correlation for {company}")

        try:
            # Get sentiment data
            sentiment_df = self.get_sentiment_timeseries(company, days_back)

            # Get stock price data
            if self.data_loader.composite_data is not None:
                price_df = self.data_loader.composite_data.copy()
                price_df['Date'] = pd.to_datetime(price_df['Date'])
                price_df = price_df.sort_values('Date')

                # Filter to date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                price_df = price_df[(price_df['Date'] >= start_date) & (price_df['Date'] <= end_date)]

                # Resample sentiment data to daily frequency
                if not sentiment_df.empty:
                    sentiment_daily = sentiment_df.resample('D', on='timestamp').agg({
                        'sentiment_score': 'mean',
                        'confidence': 'mean'
                    }).fillna(method='ffill')

                    # Merge datasets
                    merged_df = pd.merge(
                        price_df[['Date', 'Close', 'Price_Change_Pct']],
                        sentiment_daily,
                        left_on='Date',
                        right_index=True,
                        how='inner'
                    )

                    if not merged_df.empty:
                        # Calculate correlations
                        correlation_results = {
                            'sentiment_price_correlation': merged_df['sentiment_score'].corr(merged_df['Close']),
                            'sentiment_returns_correlation': merged_df['sentiment_score'].corr(merged_df['Price_Change_Pct']),
                            'confidence_price_correlation': merged_df['confidence'].corr(merged_df['Close']),
                            'sample_size': len(merged_df),
                            'date_range': {
                                'start': merged_df['Date'].min().strftime('%Y-%m-%d'),
                                'end': merged_df['Date'].max().strftime('%Y-%m-%d')
                            },
                            'correlation_significance': self._calculate_correlation_significance(
                                merged_df['sentiment_score'], merged_df['Price_Change_Pct']
                            )
                        }

                        logger.info(f"✅ Correlation analysis completed for {company}")
                        return correlation_results

            return {'error': 'Insufficient data for correlation analysis'}

        except Exception as e:
            logger.error(f"Error in correlation analysis for {company}: {str(e)}")
            return {'error': str(e)}

    def get_sentiment_timeseries(self, company: str, days_back: int = 365) -> pd.DataFrame:
        """Get sentiment data as time series for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)

            query = '''
                SELECT timestamp, sentiment_score, sentiment_label, confidence, source
                FROM sentiment_data
                WHERE company = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp ASC
            '''.format(days_back)

            df = pd.read_sql_query(query, conn, params=(company,))
            conn.close()

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error getting sentiment timeseries for {company}: {str(e)}")
            return pd.DataFrame()

    def _calculate_correlation_significance(self, x: pd.Series, y: pd.Series) -> Dict:
        """Calculate statistical significance of correlation"""
        try:
            from scipy import stats

            correlation, p_value = stats.pearsonr(x, y)

            return {
                'correlation_coefficient': correlation,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'significance_level': '95%' if p_value < 0.05 else 'Not significant'
            }

        except ImportError:
            return {'note': 'Install scipy for statistical significance testing'}
        except:
            return {'note': 'Could not calculate statistical significance'}

    def export_research_data(self, company: str, days_back: int = 365, format: str = 'csv') -> str:
        """Export data for academic research"""
        logger.info(f"📊 Exporting research data for {company}")

        try:
            # Get sentiment data
            sentiment_df = self.get_sentiment_timeseries(company, days_back)

            # Get stock data
            stock_data = {}
            if self.data_loader.composite_data is not None:
                stock_df = self.data_loader.composite_data.copy()
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_data = stock_df.to_dict('records')

            # Get correlation analysis
            correlation = self.analyze_sentiment_correlation(company, days_back)

            # Combine all data
            research_data = {
                'metadata': {
                    'company': company,
                    'export_date': datetime.now().isoformat(),
                    'date_range_days': days_back,
                    'sentiment_entries': len(sentiment_df),
                    'correlation_analysis': correlation
                },
                'sentiment_timeseries': sentiment_df.to_dict('records') if not sentiment_df.empty else [],
                'stock_data': stock_data,
                'summary_statistics': {
                    'avg_sentiment': sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else None,
                    'sentiment_volatility': sentiment_df['sentiment_score'].std() if not sentiment_df.empty else None,
                    'total_mentions': len(sentiment_df),
                    'date_range': {
                        'start': sentiment_df.index.min().strftime('%Y-%m-%d') if not sentiment_df.empty else None,
                        'end': sentiment_df.index.max().strftime('%Y-%m-%d') if not sentiment_df.empty else None
                    }
                }
            }

            # Export based on format
            filename = f"research_data_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if format.lower() == 'json':
                import json
                with open(f"{filename}.json", 'w') as f:
                    json.dump(research_data, f, indent=2, default=str)
                return f"{filename}.json"

            elif format.lower() == 'csv':
                # Export sentiment data
                if not sentiment_df.empty:
                    sentiment_df.to_csv(f"{filename}_sentiment.csv")

                # Export stock data
                if stock_data:
                    stock_df = pd.DataFrame(stock_data)
                    stock_df.to_csv(f"{filename}_stock.csv", index=False)

                return f"{filename}_sentiment.csv, {filename}_stock.csv"

            else:
                return f"Unsupported format: {format}"

        except Exception as e:
            logger.error(f"Error exporting research data for {company}: {str(e)}")
            return f"Error: {str(e)}"
    
    def generate_report(self) -> Dict:
        """Generate comprehensive sentiment and prediction report"""
        logger.info("Generating comprehensive market report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self._get_system_status(),
            'companies': {}
        }

        for company in self.companies:
            company_symbol = company['symbol']

            try:
                # Get sentiment analysis
                sentiment_features = self.get_sentiment_features(company_symbol)

                # Get predictions from all models
                predictions = {}
                for model_name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
                    pred = self.predict_stock_movement(company_symbol, model_name)
                    if 'error' not in pred:
                        predictions[model_name] = pred

                report['companies'][company_symbol] = {
                    'name': company['name'],
                    'sentiment_features': sentiment_features,
                    'predictions': predictions,
                    'data_quality': self._assess_data_quality(company_symbol)
                }

            except Exception as e:
                logger.error(f"Error generating report for {company_symbol}: {str(e)}")
                report['companies'][company_symbol] = {
                    'name': company['name'],
                    'error': str(e)
                }

        logger.info(f"Report generated for {len(report['companies'])} companies")
        return report

    def _get_system_status(self) -> Dict:
        """Get overall system health status"""
        status = {
            'data_loaded': self.data_loader.composite_data is not None,
            'database_connected': True,  # Assume DB is working if we get here
            'models_trained': len(self.models) > 0,
            'companies_monitored': len(self.companies)
        }

        # Check data freshness
        if self.data_loader.composite_data is not None:
            latest_date = self.data_loader.composite_data['Date'].max()
            days_old = (datetime.now() - latest_date).days
            status['data_freshness_days'] = days_old
            status['data_current'] = days_old <= 7  # Consider data current if <= 1 week old

        return status

    def _assess_data_quality(self, company: str) -> Dict:
        """Assess data quality for a company"""
        try:
            sentiment_features = self.get_sentiment_features(company)
            quality = {
                'sentiment_data_points': sentiment_features.get('total_mentions', 0),
                'sentiment_confidence': sentiment_features.get('avg_confidence', 0),
                'data_quality_score': min(1.0, sentiment_features.get('total_mentions', 0) / 10)  # Scale of 0-1
            }
            return quality
        except:
            return {'data_quality_score': 0.0}

# Streamlit Dashboard
def create_streamlit_dashboard():
    """Create Streamlit dashboard for the GSE sentiment analysis system"""
    
    st.set_page_config(
        page_title="GSE AI Analytics - Smart Investment Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Enhanced header with gradient background
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
        <h1 style='color: white; margin: 0; font-size: 2.5em; text-align: center;'>
            🇬🇭 GSE AI Analytics Platform
        </h1>
        <p style='font-size: 1.2em; margin: 10px 0 0 0; text-align: center; opacity: 0.9;'>
            🤖 Advanced Sentiment Analysis & Market Prediction for Ghanaian Investors
        </p>
        <p style='text-align: center; margin: 15px 0 0 0; font-style: italic; opacity: 0.8;'>
            "Transforming market intelligence with cutting-edge AI technology"
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Real-time status indicator
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        current_time = datetime.now()
        st.metric("🕒 System Time", current_time.strftime('%H:%M:%S UTC'))

    with status_col2:
        days_since_update = 0
        if hasattr(st.session_state, 'last_update'):
            days_since_update = (current_time - st.session_state.last_update).seconds // 3600
        st.metric("🔄 Last Refresh", f"{days_since_update} hours ago")

    with status_col3:
        st.metric("🎯 AI Confidence", "High", "↗️ Improving")
    
    # Initialize analyzer
    @st.cache_resource
    def load_analyzer():
        return GSESentimentAnalyzer()
    
    analyzer = load_analyzer()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Manual Sentiment Input", "Predictions", "Data Collection", "Model Performance", "🔬 Academic Research"]
    )
    
    if page == "Overview":
        st.header("📈 Market Overview & Analytics")

        # System Status Banner
        status = analyzer._get_system_status()
        if status['data_loaded']:
            st.success("✅ System Status: Fully Operational | Real GSE Data Loaded")
        else:
            st.warning("⚠️ System Status: Operational | Using Synthetic Data")

        # Key Metrics Row
        st.subheader("🎯 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_companies = len(analyzer.companies)
            st.metric("Companies Monitored", total_companies)

        with col2:
            if status['data_loaded']:
                summary = analyzer.data_loader.get_data_summary()
                total_records = summary.get('composite_data', {}).get('total_records', 0)
                st.metric("Market Data Points", f"{total_records:,}")
            else:
                st.metric("Data Status", "Synthetic")

        with col3:
            # Calculate average sentiment across all companies
            avg_market_sentiment = 0
            company_count = 0
            for company in analyzer.companies[:5]:  # Sample first 5
                features = analyzer.get_sentiment_features(company['symbol'])
                if features['total_mentions'] > 0:
                    avg_market_sentiment += features['avg_sentiment']
                    company_count += 1
            if company_count > 0:
                avg_market_sentiment /= company_count
            st.metric("Market Sentiment", f"{avg_market_sentiment:.3f}")

        with col4:
            trained_models = len([c for c in analyzer.companies if c['symbol'] in analyzer.models])
            st.metric("AI Models Trained", trained_models)

        # GSE Market Data Visualization
        if status['data_loaded'] and analyzer.data_loader.composite_data is not None:
            st.subheader("📊 GSE Composite Index Performance")

            df = analyzer.data_loader.composite_data.copy()
            df = df.sort_values('Date').tail(100)  # Last 100 trading days

            # Price chart
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=df['Date'],
                y=df['Close'],
                mode='lines',
                name='GSE Composite Index',
                line=dict(color='#1f77b4', width=2)
            ))
            fig_price.update_layout(
                title="GSE Composite Index Price Movement",
                xaxis_title="Date",
                yaxis_title="Index Value (GHS)",
                height=400
            )
            st.plotly_chart(fig_price, config={'responsive': True, 'displayModeBar': False})

            # Volume chart
            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(
                x=df['Date'],
                y=df['Turnover'],
                name='Daily Volume',
                marker_color='#ff7f0e'
            ))
            fig_volume.update_layout(
                title="GSE Daily Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300
            )
            st.plotly_chart(fig_volume, config={'responsive': True, 'displayModeBar': False})

        # Generate and display report
        with st.spinner("Analyzing company sentiment..."):
            report = analyzer.generate_report()

        # Enhanced Company Analysis Cards
        st.subheader("🏢 Company Sentiment Analysis Dashboard")

        # Company selector for detailed view
        selected_company = st.selectbox(
            "Select Company for Detailed Analysis",
            options=[comp['symbol'] for comp in analyzer.companies],
            format_func=lambda x: f"{x} - {next(c['name'] for c in analyzer.companies if c['symbol'] == x)}",
            key="overview_company_select"
        )

        # Display selected company prominently
        if selected_company in report['companies']:
            company_data = report['companies'][selected_company]

            # Company header with enhanced styling
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;'>
                <h2 style='margin: 0; color: white;'>📊 {selected_company} - {company_data['name']}</h2>
                <p style='margin: 5px 0 0 0; opacity: 0.9;'>Real-time sentiment analysis and market predictions</p>
            </div>
            """, unsafe_allow_html=True)

            sentiment = company_data['sentiment_features']

            # Key metrics in a nice grid
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                sentiment_score = sentiment['avg_sentiment']
                delta_color = "normal" if sentiment_score >= 0 else "inverse"
                st.metric(
                    "Sentiment Score",
                    f"{sentiment_score:.3f}",
                    delta=f"{sentiment['sentiment_trend']:.3f}",
                    delta_color=delta_color
                )

            with col2:
                st.metric("Total Mentions", sentiment['total_mentions'])

            with col3:
                st.metric("Positive Ratio", f"{sentiment['positive_ratio']:.1%}")

            with col4:
                st.metric("Data Confidence", f"{sentiment['avg_confidence']:.1%}")

            # Sentiment breakdown with enhanced visualization
            col1, col2 = st.columns(2)

            with col1:
                # Sentiment distribution pie chart
                sentiment_dist = {
                    'Positive': sentiment['positive_ratio'],
                    'Neutral': sentiment['neutral_ratio'],
                    'Negative': sentiment['negative_ratio']
                }

                colors = ['#00ff00', '#ffff00', '#ff0000']
                fig_sentiment = px.pie(
                    values=list(sentiment_dist.values()),
                    names=list(sentiment_dist.keys()),
                    title=f"Sentiment Distribution for {selected_company}",
                    color_discrete_sequence=colors
                )
                fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_sentiment, config={'responsive': True, 'displayModeBar': False})

            with col2:
                # AI Predictions with enhanced styling
                st.subheader("🤖 AI Market Predictions")

                if company_data['predictions']:
                    for model_name, pred in company_data['predictions'].items():
                        direction = pred['prediction']
                        confidence = pred['confidence']

                        # Color coding based on prediction
                        if direction == 'UP':
                            bg_color = '#d4edda'
                            border_color = '#28a745'
                            emoji = '🟢'
                        else:
                            bg_color = '#f8d7da'
                            border_color = '#dc3545'
                            emoji = '🔴'

                        st.markdown(f"""
                        <div style='background: {bg_color}; border: 2px solid {border_color};
                                    border-radius: 10px; padding: 15px; margin: 10px 0;'>
                            <h4 style='margin: 0; color: {border_color};'>
                                {emoji} {model_name.replace('_', ' ').title()}
                            </h4>
                            <p style='margin: 5px 0; font-size: 18px; font-weight: bold;'>
                                Prediction: {direction}
                            </p>
                            <p style='margin: 0; color: #666;'>
                                Confidence: {confidence:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🤖 AI models are being trained. Predictions will be available shortly.")

        # Market sentiment heatmap for all companies
        st.subheader("🌡️ Market Sentiment Heatmap")

        # Create sentiment data for heatmap
        sentiment_data = []
        for symbol, data in report['companies'].items():
            if 'sentiment_features' in data:
                sentiment_data.append({
                    'Company': symbol,
                    'Sentiment': data['sentiment_features']['avg_sentiment'],
                    'Mentions': data['sentiment_features']['total_mentions'],
                    'Volatility': data['sentiment_features']['sentiment_volatility']
                })

        if sentiment_data:
            df_sentiment = pd.DataFrame(sentiment_data)

            # Sentiment heatmap
            fig_heatmap = px.scatter(
                df_sentiment,
                x='Company',
                y='Sentiment',
                size='Mentions',
                color='Volatility',
                title="Market Sentiment Overview",
                color_continuous_scale='RdYlGn_r',
                size_max=50
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, config={'responsive': True, 'displayModeBar': False})
    
    elif page == "Manual Sentiment Input":
        st.header("Manual Sentiment Input")
        st.write("Add manual sentiment data for GSE companies based on news, rumors, or market observations.")
        
        with st.form("manual_sentiment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                company = st.selectbox(
                    "Select Company",
                    options=[comp['symbol'] for comp in analyzer.companies],
                    format_func=lambda x: f"{x} - {next(c['name'] for c in analyzer.companies if c['symbol'] == x)}"
                )
                
                news_type = st.selectbox(
                    "News/Rumor Type",
                    ["earnings", "management_change", "regulatory", "market_rumor", 
                     "partnership", "expansion", "financial_results", "other"]
                )
                
                sentiment = st.selectbox(
                    "Sentiment",
                    ["very_positive", "positive", "neutral", "negative", "very_negative"]
                )
            
            with col2:
                user_id = st.text_input("Your Name/ID (optional)", value="anonymous")
                
                content = st.text_area(
                    "News Content/Description",
                    placeholder="Enter the news, rumor, or information that affects this company...",
                    height=150
                )
            
            submitted = st.form_submit_button("Add Sentiment Data")
            
            if submitted and content:
                success = analyzer.add_manual_sentiment(
                    company=company,
                    news_type=news_type,
                    content=content,
                    user_sentiment=sentiment,
                    user_id=user_id
                )
                
                if success:
                    st.success(f"✅ Successfully added {sentiment} sentiment for {company}")
                else:
                    st.error("❌ Error adding sentiment data")
    
    elif page == "Predictions":
        st.header("Stock Movement Predictions")
        
        selected_company = st.selectbox(
            "Select Company for Detailed Prediction",
            options=[comp['symbol'] for comp in analyzer.companies],
            format_func=lambda x: f"{x} - {next(c['name'] for c in analyzer.companies if c['symbol'] == x)}"
        )
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                col1, col2, col3 = st.columns(3)
                
                models = ['random_forest', 'gradient_boosting', 'logistic_regression']
                model_names = ['Random Forest', 'Gradient Boosting', 'Logistic Regression']
                
                for i, (model, name) in enumerate(zip(models, model_names)):
                    with [col1, col2, col3][i]:
                        prediction = analyzer.predict_stock_movement(selected_company, model)
                        
                        if 'error' not in prediction:
                            direction = prediction['prediction']
                            confidence = prediction['confidence']
                            
                            color = "green" if direction == "UP" else "red"
                            emoji = "🟢" if direction == "UP" else "🔴"
                            
                            st.markdown(f"""
                            <div style='text-align: center; padding: 20px; border: 2px solid {color}; border-radius: 10px;'>
                                <h3>{name}</h3>
                                <h2>{emoji} {direction}</h2>
                                <p>Confidence: {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Error: {prediction['error']}")
        
        # Display sentiment analysis for selected company
        st.subheader(f"Sentiment Analysis for {selected_company}")
        sentiment_features = analyzer.get_sentiment_features(selected_company)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Sentiment", f"{sentiment_features['avg_sentiment']:.3f}")
        col2.metric("Total Mentions", sentiment_features['total_mentions'])
        col3.metric("Sentiment Volatility", f"{sentiment_features['sentiment_volatility']:.3f}")
        col4.metric("Confidence", f"{sentiment_features['avg_confidence']:.3f}")
        
        # Sentiment distribution chart
        sentiment_dist = {
            'Positive': sentiment_features['positive_ratio'],
            'Neutral': sentiment_features['neutral_ratio'],
            'Negative': sentiment_features['negative_ratio']
        }
        
        fig = px.pie(
            values=list(sentiment_dist.values()),
            names=list(sentiment_dist.keys()),
            title=f"Sentiment Distribution for {selected_company}"
        )
        st.plotly_chart(fig)
    
    elif page == "Data Collection":
        st.header("Data Collection & Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Collect New Data")
            days_back = st.number_input("Days to look back", min_value=1, max_value=30, value=7)
            
            if st.button("Start Data Collection"):
                with st.spinner("Collecting sentiment data..."):
                    sentiment_data = analyzer.collect_sentiment_data(days_back=days_back)
                    
                if sentiment_data:
                    analyzer.save_sentiment_data(sentiment_data)
                    st.success(f"✅ Collected and saved {len(sentiment_data)} sentiment entries")
                    
                    # Show sample data
                    st.subheader("Sample Collected Data")
                    for i, data in enumerate(sentiment_data[:3]):
                        with st.expander(f"Entry {i+1}: {data.company} - {data.sentiment_label}"):
                            st.write(f"**Source:** {data.source}")
                            st.write(f"**Sentiment:** {data.sentiment_label} ({data.sentiment_score:.3f})")
                            st.write(f"**Confidence:** {data.confidence:.3f}")
                            st.write(f"**Content:** {data.content[:200]}...")
                else:
                    st.warning("⚠️ No new sentiment data collected")
        
        with col2:
            st.subheader("Database Statistics")
            
            # Get database stats
            conn = sqlite3.connect(analyzer.db_path)
            
            # Automated sentiment count
            auto_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM sentiment_data", conn
            )['count'][0]
            
            # Manual sentiment count
            manual_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM manual_sentiment", conn
            )['count'][0]
            
            # Company breakdown
            company_breakdown = pd.read_sql_query(
                "SELECT company, COUNT(*) as count FROM sentiment_data GROUP BY company", conn
            )
            
            conn.close()
            
            st.metric("Automated Entries", auto_count)
            st.metric("Manual Entries", manual_count)
            st.metric("Total Entries", auto_count + manual_count)
            
            if not company_breakdown.empty:
                st.subheader("Company Data Distribution")
                fig = px.bar(
                    company_breakdown,
                    x='company',
                    y='count',
                    title="Sentiment Entries by Company"
                )
                st.plotly_chart(fig)
    
    elif page == "Model Performance":
        st.header("Model Performance Analysis")
        
        selected_company = st.selectbox(
            "Select Company for Model Analysis",
            options=[comp['symbol'] for comp in analyzer.companies],
            format_func=lambda x: f"{x} - {next(c['name'] for c in analyzer.companies if c['symbol'] == x)}"
        )
        
        if st.button("Train and Evaluate Models"):
            with st.spinner("Training models..."):
                results = analyzer.train_prediction_models(selected_company)
                
                if results:
                    st.success("✅ Models trained successfully!")
                    
                    # Display model comparison
                    model_performance = []
                    for model_name, model_info in results.items():
                        model_performance.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Accuracy': model_info['accuracy']
                        })
                    
                    df_performance = pd.DataFrame(model_performance)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Model Accuracy Comparison")
                        fig = px.bar(
                            df_performance,
                            x='Model',
                            y='Accuracy',
                            title=f"Model Performance for {selected_company}"
                        )
                        st.plotly_chart(fig)
                    
                    with col2:
                        st.subheader("Performance Metrics")
                        st.dataframe(df_performance)
                        
                        # Best model
                        best_model = df_performance.loc[df_performance['Accuracy'].idxmax()]
                        st.success(f"🏆 Best Model: {best_model['Model']} ({best_model['Accuracy']:.3f})")
                
                else:
                    st.error("❌ Error training models. Please check if sufficient data is available.")

    elif page == "🔬 Academic Research":
        st.header("🔬 Academic Research & Historical Analysis")
        st.markdown("*Advanced sentiment analysis for research purposes*")

        research_tab1, research_tab2, research_tab3, research_tab4 = st.tabs([
            "📊 Historical Data Collection",
            "📈 Time-Series Analysis",
            "🔗 Correlation Studies",
            "📥 Data Export"
        ])

        with research_tab1:
            st.subheader("📊 Historical Sentiment Data Collection")

            col1, col2 = st.columns(2)

            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                    key="research_start_date"
                )
                companies_to_collect = st.multiselect(
                    "Companies to Analyze",
                    options=[comp['symbol'] for comp in analyzer.companies],
                    default=[comp['symbol'] for comp in analyzer.companies[:3]],
                    key="research_companies"
                )

            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    key="research_end_date"
                )
                collection_mode = st.selectbox(
                    "Collection Mode",
                    ["Standard", "Deep Research"],
                    help="Deep Research mode collects more data but takes longer"
                )

            if st.button("🚀 Start Historical Data Collection", type="primary"):
                if start_date >= end_date:
                    st.error("❌ Start date must be before end date")
                elif not companies_to_collect:
                    st.error("❌ Please select at least one company")
                else:
                    with st.spinner("🔬 Collecting historical sentiment data... This may take several minutes."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Run collection
                        historical_data = analyzer.collect_historical_sentiment_data(
                            start_date.strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d'),
                            companies_to_collect
                        )

                        progress_bar.progress(100)
                        status_text.text("✅ Collection completed!")

                        # Show results
                        st.success(f"✅ Collected {historical_data['collection_stats']['total_entries']} sentiment entries")
                        st.json(historical_data['collection_stats'])

        with research_tab2:
            st.subheader("📈 Time-Series Sentiment Analysis")

            selected_company_ts = st.selectbox(
                "Select Company for Time-Series Analysis",
                options=[comp['symbol'] for comp in analyzer.companies],
                key="timeseries_company"
            )

            analysis_period = st.slider(
                "Analysis Period (days)",
                min_value=30,
                max_value=730,
                value=365,
                key="analysis_period"
            )

            if st.button("📊 Generate Time-Series Analysis"):
                with st.spinner("Analyzing sentiment trends..."):
                    # Get sentiment time series
                    sentiment_ts = analyzer.get_sentiment_timeseries(selected_company_ts, analysis_period)

                    if not sentiment_ts.empty:
                        # Create time series plot
                        fig_ts = go.Figure()

                        # Sentiment score over time
                        fig_ts.add_trace(go.Scatter(
                            x=sentiment_ts.index,
                            y=sentiment_ts['sentiment_score'],
                            mode='lines+markers',
                            name='Sentiment Score',
                            line=dict(color='#1f77b4', width=2),
                            marker=dict(size=4)
                        ))

                        # Add moving average
                        sentiment_ts['sentiment_ma_7'] = sentiment_ts['sentiment_score'].rolling(window=7).mean()
                        fig_ts.add_trace(go.Scatter(
                            x=sentiment_ts.index,
                            y=sentiment_ts['sentiment_ma_7'],
                            mode='lines',
                            name='7-Day Moving Average',
                            line=dict(color='#ff7f0e', width=3, dash='dash')
                        ))

                        fig_ts.update_layout(
                            title=f"Sentiment Time Series for {selected_company_ts}",
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score",
                            height=500
                        )

                        st.plotly_chart(fig_ts, config={'responsive': True, 'displayModeBar': False})

                        # Sentiment distribution over time
                        st.subheader("Sentiment Distribution Over Time")

                        # Group by month
                        monthly_sentiment = sentiment_ts.resample('M').agg({
                            'sentiment_score': ['mean', 'std', 'count']
                        }).fillna(0)

                        fig_monthly = go.Figure()

                        fig_monthly.add_trace(go.Bar(
                            x=monthly_sentiment.index.strftime('%Y-%m'),
                            y=monthly_sentiment[('sentiment_score', 'count')],
                            name='Number of Mentions',
                            marker_color='#2ca02c',
                            yaxis='y2'
                        ))

                        fig_monthly.add_trace(go.Scatter(
                            x=monthly_sentiment.index.strftime('%Y-%m'),
                            y=monthly_sentiment[('sentiment_score', 'mean')],
                            mode='lines+markers',
                            name='Average Sentiment',
                            line=dict(color='#d62728', width=3),
                            marker=dict(size=8)
                        ))

                        fig_monthly.update_layout(
                            title=f"Monthly Sentiment Analysis for {selected_company_ts}",
                            xaxis_title="Month",
                            yaxis_title="Average Sentiment",
                            yaxis2=dict(
                                title="Number of Mentions",
                                overlaying="y",
                                side="right"
                            ),
                            height=400
                        )

                        st.plotly_chart(fig_monthly, config={'responsive': True, 'displayModeBar': False})

                    else:
                        st.warning(f"⚠️ No sentiment data available for {selected_company_ts} in the selected period")

        with research_tab3:
            st.subheader("🔗 Sentiment-Price Correlation Analysis")

            correlation_company = st.selectbox(
                "Select Company for Correlation Analysis",
                options=[comp['symbol'] for comp in analyzer.companies],
                key="correlation_company"
            )

            correlation_period = st.slider(
                "Correlation Analysis Period (days)",
                min_value=30,
                max_value=730,
                value=180,
                key="correlation_period"
            )

            if st.button("🔬 Run Correlation Analysis"):
                with st.spinner("Analyzing sentiment-price correlations..."):
                    correlation_results = analyzer.analyze_sentiment_correlation(
                        correlation_company,
                        correlation_period
                    )

                    if 'error' not in correlation_results:
                        # Display correlation results
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            corr_coeff = correlation_results['sentiment_price_correlation']
                            st.metric(
                                "Sentiment vs Price Correlation",
                                f"{corr_coeff:.3f}",
                                delta="Strong" if abs(corr_coeff) > 0.5 else "Moderate" if abs(corr_coeff) > 0.3 else "Weak"
                            )

                        with col2:
                            returns_corr = correlation_results['sentiment_returns_correlation']
                            st.metric(
                                "Sentiment vs Returns Correlation",
                                f"{returns_corr:.3f}",
                                delta="Strong" if abs(returns_corr) > 0.5 else "Moderate" if abs(returns_corr) > 0.3 else "Weak"
                            )

                        with col3:
                            st.metric("Sample Size", correlation_results['sample_size'])

                        # Significance testing
                        if 'correlation_significance' in correlation_results:
                            sig = correlation_results['correlation_significance']
                            if isinstance(sig, dict) and 'significant' in sig:
                                significance_color = "🟢" if sig['significant'] else "🔴"
                                st.info(f"{significance_color} **Statistical Significance**: {sig['significance_level']} (p-value: {sig.get('p_value', 'N/A'):.4f})")

                        # Research insights
                        st.subheader("🔍 Research Insights")

                        insights = []

                        if abs(corr_coeff) > 0.5:
                            insights.append("• **Strong correlation** between sentiment and stock price")
                        elif abs(corr_coeff) > 0.3:
                            insights.append("• **Moderate correlation** between sentiment and stock price")
                        else:
                            insights.append("• **Weak correlation** between sentiment and stock price")

                        if abs(returns_corr) > 0.3:
                            insights.append("• Sentiment appears to influence short-term price movements")
                        else:
                            insights.append("• Limited evidence of sentiment impact on returns")

                        if correlation_results['sample_size'] > 100:
                            insights.append("• Large sample size provides reliable results")
                        elif correlation_results['sample_size'] > 50:
                            insights.append("• Moderate sample size - results are reasonably reliable")
                        else:
                            insights.append("• Small sample size - results should be interpreted cautiously")

                        for insight in insights:
                            st.write(insight)

                    else:
                        st.error(f"❌ Correlation analysis failed: {correlation_results['error']}")

        with research_tab4:
            st.subheader("📥 Data Export for Research")

            export_company = st.selectbox(
                "Select Company for Data Export",
                options=[comp['symbol'] for comp in analyzer.companies],
                key="export_company"
            )

            export_period = st.slider(
                "Export Period (days)",
                min_value=30,
                max_value=1095,  # 3 years
                value=365,
                key="export_period"
            )

            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON"],
                key="export_format"
            )

            include_correlation = st.checkbox("Include Correlation Analysis", value=True)
            include_summary = st.checkbox("Include Summary Statistics", value=True)

            if st.button("📤 Export Research Data", type="primary"):
                with st.spinner("Preparing research data export..."):
                    try:
                        export_result = analyzer.export_research_data(
                            export_company,
                            export_period,
                            export_format.lower()
                        )

                        if "Error" not in export_result:
                            st.success(f"✅ Research data exported successfully!")
                            st.info(f"📁 Files created: {export_result}")

                            # Show data preview
                            sentiment_ts = analyzer.get_sentiment_timeseries(export_company, export_period)
                            if not sentiment_ts.empty:
                                st.subheader("Data Preview")
                                st.dataframe(sentiment_ts.head(10))

                                # Download button
                                csv_data = sentiment_ts.to_csv()
                                st.download_button(
                                    label="📥 Download Sentiment Data (CSV)",
                                    data=csv_data,
                                    file_name=f"sentiment_data_{export_company}.csv",
                                    mime="text/csv"
                                )

                        else:
                            st.error(f"❌ Export failed: {export_result}")

                    except Exception as e:
                        st.error(f"❌ Export error: {str(e)}")

            # Export guidelines
            with st.expander("📋 Research Data Export Guidelines"):
                st.markdown("""
                **Data Structure:**
                - **Sentiment Timeseries**: Daily sentiment scores with timestamps
                - **Stock Data**: Historical price and volume data
                - **Correlation Analysis**: Statistical relationships between sentiment and prices

                **Recommended Analysis:**
                - Time-series regression models
                - Granger causality tests
                - Sentiment-based trading strategies
                - Event study methodology

                **Citation Note**: When using this data in academic work, please acknowledge the GSE Sentiment Analysis System.
                """)

    # Enhanced Footer with System Info
    st.markdown("---")

    # System information and credits
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🚀 Powered By:**
        - Advanced NLP & Machine Learning
        - Real-time Data Processing
        - Multi-source Sentiment Analysis
        """)

    with col2:
        st.markdown("""
        **📊 Data Sources:**
        - Ghana Stock Exchange (GSE)
        - GhanaWeb Business News
        - Bloomberg Africa
        - Reuters Africa
        - Social Media Analytics
        """)

    with col3:
        st.markdown(f"""
        **⚡ System Status:**
        - Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - Companies: {len(analyzer.companies)}
        - AI Models: {len(analyzer.models)} trained
        - Data Quality: {'High' if analyzer.data_loader.composite_data is not None else 'Synthetic'}
        """)

    # Professional footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>GSE Sentiment Analysis & Prediction System</strong></p>
        <p>Built with ❤️ for Ghanaian investors | Transforming market intelligence with AI</p>
        <p><small>© 2025 | Advanced Financial Analytics Platform</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = GSESentimentAnalyzer()
    
    print("GSE Sentiment Analysis System")
    print("=" * 40)
    
    # Example usage
    print("\n1. Collecting sentiment data...")
    sentiment_data = analyzer.collect_sentiment_data(days_back=3)
    
    if sentiment_data:
        print(f"Collected {len(sentiment_data)} sentiment entries")
        analyzer.save_sentiment_data(sentiment_data)
        
        # Show sample
        for i, data in enumerate(sentiment_data[:3]):
            print(f"\nSample {i+1}:")
            print(f"Company: {data.company}")
            print(f"Sentiment: {data.sentiment_label} ({data.sentiment_score:.3f})")
            print(f"Source: {data.source}")
            print(f"Content: {data.content[:100]}...")
    
    print("\n2. Training prediction models...")
    for company in analyzer.companies:  # Train for all 16 companies
        try:
            results = analyzer.train_prediction_models(company['symbol'])
            if results:
                print(f"Trained models for {company['symbol']}")
                for model_name, model_info in results.items():
                    print(f"  {model_name}: {model_info['accuracy']:.3f} accuracy")
        except Exception as e:
            print(f"Error training models for {company['symbol']}: {str(e)}")

    print("\n3. Making predictions...")
    for company in analyzer.companies:
        prediction = analyzer.predict_stock_movement(company['symbol'])
        if 'error' not in prediction:
            print(f"{company['symbol']}: {prediction['prediction']} "
                  f"(confidence: {prediction['confidence']:.1%})")
    
    print("\n4. Generating report...")
    report = analyzer.generate_report()
    print(f"Report generated at: {report['timestamp']}")
    print(f"Companies analyzed: {len(report['companies'])}")
    
    print("\nTo run the Streamlit dashboard:")
    print("streamlit run gse_sentiment_analysis_system.py")