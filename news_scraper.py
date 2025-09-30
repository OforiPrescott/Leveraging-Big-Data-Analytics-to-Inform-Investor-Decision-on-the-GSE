"""
News Scraper for GSE Sentiment Analysis
Advanced web scraping for multiple Ghanaian and international news sources
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import feedparser
from urllib.parse import urljoin, urlparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import hashlib

logger = logging.getLogger(__name__)

class NewsScraper:
    """Advanced news scraper for GSE-related content"""
    
    def __init__(self, delay_range: Tuple[int, int] = (1, 3)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Define news sources with specific scraping configurations
        self.news_sources = {
            'ghanaweb': {
                'base_url': 'https://www.ghanaweb.com',
                'business_section': '/GhanaHomePage/business/',
                'search_url': '/GhanaHomePage/business/search.php?q={}',
                'article_selector': 'article, .article-content, .story-content',
                'title_selector': 'h1, .article-title, .story-title',
                'content_selector': '.article-body, .story-body, p',
                'date_selector': '.article-date, .story-date, .date'
            },
            'myjoyonline': {
                'base_url': 'https://www.myjoyonline.com',
                'business_section': '/business/',
                'search_url': '/search/{}',
                'article_selector': 'article, .post-content',
                'title_selector': 'h1, .entry-title',
                'content_selector': '.entry-content, .post-body, p',
                'date_selector': '.post-date, .entry-date'
            },
            'graphic_online': {
                'base_url': 'https://www.graphic.com.gh',
                'business_section': '/business/',
                'search_url': '/search?q={}',
                'article_selector': 'article, .article-content',
                'title_selector': 'h1, .article-title',
                'content_selector': '.article-body, p',
                'date_selector': '.article-date'
            },
            'business_ghana': {
                'base_url': 'https://businessghana.com',
                'business_section': '/',
                'search_url': '/search/{}',
                'article_selector': 'article, .post-content',
                'title_selector': 'h1, .post-title',
                'content_selector': '.post-body, .entry-content, p',
                'date_selector': '.post-date'
            },
            'citi_newsroom': {
                'base_url': 'https://citinewsroom.com',
                'business_section': '/category/business/',
                'search_url': '/search/{}',
                'article_selector': 'article, .entry-content',
                'title_selector': 'h1, .entry-title',
                'content_selector': '.entry-content, p',
                'date_selector': '.entry-date'
            }
        }
        
        # GSE-related keywords for filtering relevant content
        self.gse_keywords = [
            'ghana stock exchange', 'gse', 'stock market', 'shares', 'equity',
            'dividend', 'ipo', 'listing', 'trading', 'market cap', 'broker',
            'securities', 'investment', 'portfolio', 'bull market', 'bear market'
        ]
        
        # Company-specific keywords (you can expand this)
        self.company_keywords = {
            'MTN': ['mtn ghana', 'mtn', 'mobile telecommunication', 'telco'],
            'GCB': ['gcb bank', 'ghana commercial bank', 'gcb'],
            'EGH': ['ecobank ghana', 'ecobank', 'eco bank'],
            'TOTAL': ['totalenergies', 'total', 'petroleum'],
            'ACCESS': ['access bank', 'access bank ghana'],
            'CAL': ['cal bank', 'cal'],
            'SCB': ['standard chartered', 'stanchart'],
            'AGA': ['anglogold ashanti', 'anglogold', 'gold mining']
        }
    
    def _make_request(self, url: str, timeout: int = 15) -> Optional[requests.Response]:
        """Make HTTP request with error handling and rate limiting"""
        try:
            # Random delay to be respectful
            time.sleep(random.uniform(*self.delay_range))
            
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {str(e)}")
            return None
    
    def scrape_article_content(self, url: str, source_config: Dict) -> Dict:
        """Scrape content from a single article"""
        try:
            response = self._make_request(url)
            if not response:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_selectors = source_config['title_selector'].split(', ')
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            # Extract content
            content = ""
            content_selectors = source_config['content_selector'].split(', ')
            for selector in content_selectors:
                content_elems = soup.select(selector)
                if content_elems:
                    content = ' '.join([elem.get_text().strip() for elem in content_elems])
                    break
            
            # Extract date
            date_published = None
            date_selectors = source_config['date_selector'].split(', ')
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_text = date_elem.get_text().strip()
                    date_published = self._parse_date(date_text)
                    break
            
            # Clean content
            content = re.sub(r'\s+', ' ', content).strip()
            title = re.sub(r'\s+', ' ', title).strip()
            
            # Check if content is relevant to GSE
            relevance_score = self._calculate_relevance(f"{title} {content}")
            
            if len(content) < 100:  # Skip very short articles
                return {}
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'date_published': date_published,
                'relevance_score': relevance_score,
                'scraped_at': datetime.now(),
                'source': urlparse(url).netloc
            }
            
        except Exception as e:
            logger.error(f"Error scraping article {url}: {str(e)}")
            return {}
    
    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """Parse date from various formats"""
        date_patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
            r'(\w+)\s+(\d{1,2}),\s+(\d{4})',       # Month DD, YYYY
            r'(\d{1,2})\s+(\w+)\s+(\d{4})',        # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    # This is a simplified parser - you might want to use dateutil.parser
                    groups = match.groups()
                    if len(groups) == 3:
                        if groups[0].isdigit() and len(groups[0]) == 4:  # YYYY format
                            return datetime(int(groups[0]), int(groups[1]), int(groups[2]))
                        elif groups[2].isdigit() and len(groups[2]) == 4:  # DD/MM/YYYY format
                            return datetime(int(groups[2]), int(groups[1]), int(groups[0]))
                except:
                    continue
        
        return None
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score based on GSE and company keywords"""
        text_lower = text.lower()
        
        # Base relevance for GSE keywords
        gse_score = sum(1 for keyword in self.gse_keywords if keyword in text_lower)
        
        # Company-specific relevance
        company_score = 0
        for company, keywords in self.company_keywords.items():
            company_score += sum(1 for keyword in keywords if keyword in text_lower)
        
        # Calculate final score (0-1 scale)
        total_keywords = len(self.gse_keywords) + sum(len(keywords) for keywords in self.company_keywords.values())
        relevance = (gse_score * 2 + company_score) / (total_keywords * 0.1)  # Weight GSE keywords more
        
        return min(1.0, relevance)
    
    def search_news_articles(self, query: str, source: str, max_results: int = 20) -> List[str]:
        """Search for news articles on a specific source"""
        try:
            if source not in self.news_sources:
                logger.warning(f"Unknown news source: {source}")
                return []
            
            source_config = self.news_sources[source]
            base_url = source_config['base_url']
            
            # Try different search approaches
            article_urls = []
            
            # Method 1: Search URL if available
            if 'search_url' in source_config:
                search_url = base_url + source_config['search_url'].format(query.replace(' ', '+'))
                urls_from_search = self._extract_article_urls_from_page(search_url, base_url)
                article_urls.extend(urls_from_search)
            
            # Method 2: Browse business section
            if len(article_urls) < max_results and 'business_section' in source_config:
                business_url = base_url + source_config['business_section']
                urls_from_business = self._extract_article_urls_from_page(business_url, base_url)
                article_urls.extend(urls_from_business)
            
            # Remove duplicates and limit results
            article_urls = list(dict.fromkeys(article_urls))[:max_results]
            
            logger.info(f"Found {len(article_urls)} article URLs from {source}")
            return article_urls
            
        except Exception as e:
            logger.error(f"Error searching {source} for '{query}': {str(e)}")
            return []
    
    def _extract_article_urls_from_page(self, page_url: str, base_url: str) -> List[str]:
        """Extract article URLs from a news page"""
        try:
            response = self._make_request(page_url)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            article_urls = []
            
            # Common patterns for article links
            link_selectors = [
                'a[href*="/article/"]',
                'a[href*="/news/"]',
                'a[href*="/story/"]',
                'a[href*="/post/"]',
                'a[href*="/business/"]',
                'article a',
                '.article-link',
                '.news-link'
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        if not href.startswith('http'):
                            href = urljoin(base_url, href)
                        
                        # Filter for actual article URLs
                        if self._is_article_url(href):
                            article_urls.append(href)
            
            return list(dict.fromkeys(article_urls))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting URLs from {page_url}: {str(e)}")
            return []
    
    def _is_article_url(self, url: str) -> bool:
        """Check if URL looks like an article URL"""
        article_patterns = [
            r'/article/',
            r'/news/',
            r'/story/',
            r'/post/',
            r'/\d{4}/\d{2}/',  # Date-based URLs
            r'-\d+\.html?$',   # ID-based URLs
        ]
        
        url_lower = url.lower()
        
        # Exclude common non-article URLs
        exclude_patterns = [
            'category', 'tag', 'search', 'page', 'author',
            'archive', 'feed', 'rss', 'xml', 'sitemap'
        ]
        
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False
        
        # Check for article patterns
        for pattern in article_patterns:
            if re.search(pattern, url_lower):
                return True
        
        return False
    
    def scrape_multiple_sources(self, query: str, max_articles_per_source: int = 10) -> List[Dict]:
        """Scrape articles from multiple sources concurrently"""
        all_articles = []
        
        # Collect URLs from all sources
        all_urls = []
        for source_name in self.news_sources.keys():
            urls = self.search_news_articles(query, source_name, max_articles_per_source)
            all_urls.extend(urls)
        
        if not all_urls:
            logger.warning("No article URLs found")
            return []
        
        logger.info(f"Scraping {len(all_urls)} articles...")
        
        # Scrape articles concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Create futures for each URL
            future_to_url = {}
            for url in all_urls:
                source_name = self._get_source_name_from_url(url)
                if source_name in self.news_sources:
                    source_config = self.news_sources[source_name]
                    future = executor.submit(self.scrape_article_content, url, source_config)
                    future_to_url[future] = url
            
            # Process completed futures
            for future in as_completed(future_to_url):
                try:
                    article = future.result(timeout=30)
                    if article and article.get('relevance_score', 0) > 0.1:  # Filter by relevance
                        all_articles.append(article)
                except Exception as e:
                    url = future_to_url[future]
                    logger.error(f"Error processing {url}: {str(e)}")
        
        # Sort by relevance and date
        all_articles.sort(key=lambda x: (x.get('relevance_score', 0), x.get('date_published', datetime.min)), reverse=True)
        
        logger.info(f"Successfully scraped {len(all_articles)} relevant articles")
        return all_articles
    
    def _get_source_name_from_url(self, url: str) -> str:
        """Get source name from URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'ghanaweb' in domain:
            return 'ghanaweb'
        elif 'myjoyonline' in domain:
            return 'myjoyonline'
        elif 'graphic.com.gh' in domain:
            return 'graphic_online'
        elif 'businessghana' in domain:
            return 'business_ghana'
        elif 'citinewsroom' in domain:
            return 'citi_newsroom'
        else:
            return 'unknown'
    
    def scrape_rss_feeds(self, feeds: List[str]) -> List[Dict]:
        """Scrape RSS feeds for news content"""
        articles = []
        
        for feed_url in feeds:
            try:
                logger.info(f"Parsing RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    article = {
                        'url': entry.get('link', ''),
                        'title': entry.get('title', ''),
                        'content': entry.get('summary', '') or entry.get('description', ''),
                        'date_published': self._parse_rss_date(entry.get('published', '')),
                        'source': feed_url,
                        'scraped_at': datetime.now(),
                        'relevance_score': self._calculate_relevance(f"{entry.get('title', '')} {entry.get('summary', '')}")
                    }
                    
                    if article['relevance_score'] > 0.1:
                        articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {str(e)}")
        
        return articles
    
    def _parse_rss_date(self, date_str: str) -> Optional[datetime]:
        """Parse RSS date format"""
        try:
            import email.utils
            return datetime(*email.utils.parsedate(date_str)[:6])
        except:
            return None
    
    def save_articles_to_db(self, articles: List[Dict], db_path: str = "news_articles.db"):
        """Save scraped articles to SQLite database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
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
            
            for article in articles:
                # Create content hash to avoid duplicates
                content_hash = hashlib.md5(article['content'].encode()).hexdigest()
                
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO articles 
                        (url, title, content, date_published, scraped_at, source, relevance_score, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        article['url'],
                        article['title'],
                        article['content'],
                        article['date_published'],
                        article['scraped_at'],
                        article['source'],
                        article['relevance_score'],
                        content_hash
                    ))
                except sqlite3.IntegrityError:
                    continue  # Skip duplicates
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved {len(articles)} articles to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving articles to database: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scraper = NewsScraper()
    
    print("GSE News Scraper Test")
    print("=" * 30)
    
    # Test scraping
    search_terms = [
        "Ghana Stock Exchange",
        "GSE trading",
        "MTN Ghana shares",
        "GCB Bank dividend"
    ]
    
    for term in search_terms:
        print(f"\nScraping articles for: {term}")
        
        articles = scraper.scrape_multiple_sources(term, max_articles_per_source=5)
        
        if articles:
            print(f"✅ Found {len(articles)} relevant articles")
            
            # Show sample articles
            for i, article in enumerate(articles[:3]):
                print(f"\nArticle {i+1}:")
                print(f"Title: {article['title'][:80]}...")
                print(f"Source: {article['source']}")
                print(f"Relevance: {article['relevance_score']:.3f}")
                print(f"Content: {article['content'][:150]}...")
            
            # Save to database
            scraper.save_articles_to_db(articles)
        else:
            print("❌ No relevant articles found")
        
        # Small delay between searches
        time.sleep(2)
    
    print("\n✅ News scraping test completed!")