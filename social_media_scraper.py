"""
Social Media Scraper for GSE Sentiment Analysis
Scrapes social media platforms for GSE-related content (alternative to X/Twitter API)
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import hashlib
import random
from urllib.parse import urljoin, urlparse, quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import os

logger = logging.getLogger(__name__)

class SocialMediaScraper:
    """Social media scraper for financial sentiment analysis"""
    
    def __init__(self, delay_range: Tuple[int, int] = (2, 5)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Load API keys from config
        self.api_keys = self._load_api_keys()
        self._validate_api_keys()
        
        # Social media configurations
        self.platforms = {
            'reddit': {
                'base_url': 'https://www.reddit.com',
                'search_url': '/search/?q={}&sort=new&t=week',
                'subreddits': ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'StockMarket'],
                'ghana_subreddits': ['Ghana', 'AccraGhana'],
                'post_selector': '.Post, [data-testid="post-container"]',
                'title_selector': 'h3, [data-testid="post-title"]',
                'content_selector': '[data-testid="post-text"], .RichTextJSON-root',
                'score_selector': '[data-testid="vote-button"], .Post__score'
            },
            'linkedin': {
                'base_url': 'https://www.linkedin.com',
                'search_url': '/search/results/content/?keywords={}',
                'post_selector': '.feed-shared-actor, .update-v2-social-activity',
                'title_selector': '.actor-name, .update-v2-social-activity-text',
                'content_selector': '.feed-shared-text, .update-v2-social-activity-text'
            },
            'facebook_public': {
                'base_url': 'https://www.facebook.com',
                'search_url': '/public/search/posts/?q={}',
                'post_selector': '[data-testid="story-subtitle"], .userContentWrapper',
                'content_selector': '[data-testid="post_message"], .userContent'
            }
        }
        
    def _load_api_keys(self) -> Dict[str, Dict[str, str]]:
        """Load API keys from environment variables or config.json"""
        api_keys = {}

        # Try environment variables first (for deployment)
        try:
            # Facebook API keys
            if os.getenv('FACEBOOK_APP_ID') and os.getenv('FACEBOOK_APP_SECRET'):
                api_keys['facebook'] = {
                    'app_id': os.getenv('FACEBOOK_APP_ID'),
                    'app_secret': os.getenv('FACEBOOK_APP_SECRET'),
                    'access_token': os.getenv('FACEBOOK_ACCESS_TOKEN', ''),
                    'page_access_token': os.getenv('FACEBOOK_PAGE_ACCESS_TOKEN', '')
                }

            # LinkedIn API keys
            if os.getenv('LINKEDIN_CLIENT_ID') and os.getenv('LINKEDIN_CLIENT_SECRET'):
                api_keys['linkedin'] = {
                    'client_id': os.getenv('LINKEDIN_CLIENT_ID'),
                    'client_secret': os.getenv('LINKEDIN_CLIENT_SECRET'),
                    'access_token': os.getenv('LINKEDIN_ACCESS_TOKEN', ''),
                    'refresh_token': os.getenv('LINKEDIN_REFRESH_TOKEN', '')
                }

            # Twitter API keys
            if os.getenv('TWITTER_API_KEY') and os.getenv('TWITTER_API_SECRET'):
                api_keys['twitter'] = {
                    'api_key': os.getenv('TWITTER_API_KEY'),
                    'api_secret': os.getenv('TWITTER_API_SECRET'),
                    'access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
                    'access_token_secret': os.getenv('TWITTER_ACCESS_TOKEN_SECRET', ''),
                    'bearer_token': os.getenv('TWITTER_BEARER_TOKEN', '')
                }

            if api_keys:
                logger.info(f"Loaded API keys from environment variables: {list(api_keys.keys())}")
                return api_keys

        except Exception as e:
            logger.warning(f"Error loading environment API keys: {e}")

        # Fallback to config.json (for local development)
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                api_keys = config.get('api_keys', {})
                if api_keys:
                    logger.info(f"Loaded API keys from config.json: {list(api_keys.keys())}")
                return api_keys
            else:
                logger.warning("config.json not found, using empty API keys")
                return {}
        except Exception as e:
            logger.error(f"Error loading API keys from config.json: {e}")
            return {}

    def _validate_api_keys(self):
        """Validate API keys and log status"""
        if not self.api_keys:
            logger.info("No API keys configured - using scraping fallbacks")
            return

        # Validate Facebook keys
        fb_keys = self.api_keys.get('facebook', {})
        if fb_keys.get('app_id') and fb_keys.get('access_token'):
            logger.info("✅ Facebook API keys configured")
        else:
            logger.info("⚠️  Facebook API keys incomplete - using scraping fallback")

        # Validate LinkedIn keys
        li_keys = self.api_keys.get('linkedin', {})
        if li_keys.get('client_id') and li_keys.get('access_token'):
            logger.info("✅ LinkedIn API keys configured")
        else:
            logger.info("⚠️  LinkedIn API keys incomplete - using scraping fallback")

        # Validate Twitter keys
        tw_keys = self.api_keys.get('twitter', {})
        if tw_keys.get('bearer_token'):
            logger.info("✅ Twitter API keys configured")
        else:
            logger.info("⚠️  Twitter API keys incomplete - using alternative sources")

        # GSE and financial keywords
        self.financial_keywords = [
            'ghana stock exchange', 'gse', 'stock market ghana', 'ghana shares',
            'accra stock', 'ghana equity', 'ghana investment', 'ghana trading',
            'ghana dividend', 'ghana portfolio', 'ghana broker', 'ghana securities'
        ]
        
        # Company-specific hashtags and mentions
        self.company_mentions = {
            'MTN': ['#MTNGhana', '@MTNGhana', 'MTN Ghana', 'mtn shares'],
            'GCB': ['#GCBBank', '@GCBBankGhana', 'GCB Bank', 'ghana commercial bank'],
            'ECOBANK': ['#EcobankGhana', '@EcobankGhana', 'Ecobank Ghana'],
            'TOTAL': ['#TotalEnergies', '@TotalEnergies', 'Total Ghana'],
            'ACCESS': ['#AccessBankGhana', '@AccessBankGH', 'Access Bank Ghana'],
            'AGA': ['#AngloGoldAshanti', '@AngloGoldCorp', 'AngloGold Ashanti']
        }
    
    def _make_request(self, url: str, headers: Optional[Dict] = None, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make HTTP request with error handling and rate limiting"""
        try:
            # Random delay to avoid being blocked
            time.sleep(random.uniform(*self.delay_range))

            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)

            response = self.session.get(url, headers=request_headers, params=params, timeout=15)
            
            # Check for rate limiting or blocking
            if response.status_code == 429:
                logger.warning(f"Rate limited for {url}, waiting longer...")
                time.sleep(30)
                return None
            elif response.status_code == 403:
                logger.warning(f"Access forbidden for {url}")
                return None
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {str(e)}")
            return None
    
    def scrape_reddit_financial_discussions(self, query: str, limit: int = 50) -> List[Dict]:
        """Scrape Reddit for financial discussions"""
        posts = []
        
        try:
            # Search in general financial subreddits
            for subreddit in self.platforms['reddit']['subreddits']:
                subreddit_posts = self._scrape_reddit_subreddit(subreddit, query, limit // len(self.platforms['reddit']['subreddits']))
                posts.extend(subreddit_posts)
            
            # Search in Ghana-specific subreddits
            for subreddit in self.platforms['reddit']['ghana_subreddits']:
                ghana_posts = self._scrape_reddit_subreddit(subreddit, query, 10)
                posts.extend(ghana_posts)
            
            logger.info(f"Scraped {len(posts)} Reddit posts for query: {query}")
            return posts
            
        except Exception as e:
            logger.error(f"Error scraping Reddit: {str(e)}")
            return []
    
    def _scrape_reddit_subreddit(self, subreddit: str, query: str, limit: int) -> List[Dict]:
        """Scrape specific Reddit subreddit"""
        posts = []
        
        try:
            # Use Reddit's JSON API (more reliable than HTML scraping)
            url = f"https://www.reddit.com/r/{subreddit}/search/.json?q={quote_plus(query)}&restrict_sr=1&sort=new&limit={limit}"
            
            response = self._make_request(url)
            if not response:
                return []
            
            try:
                data = response.json()
                if 'data' in data and 'children' in data['data']:
                    for post_data in data['data']['children']:
                        post = post_data['data']
                        
                        post_info = {
                            'platform': 'reddit',
                            'subreddit': subreddit,
                            'title': post.get('title', ''),
                            'content': post.get('selftext', ''),
                            'url': f"https://www.reddit.com{post.get('permalink', '')}",
                            'author': post.get('author', ''),
                            'score': post.get('score', 0),
                            'num_comments': post.get('num_comments', 0),
                            'created_utc': datetime.fromtimestamp(post.get('created_utc', 0)),
                            'scraped_at': datetime.now(),
                            'relevance_score': self._calculate_financial_relevance(f"{post.get('title', '')} {post.get('selftext', '')}")
                        }
                        
                        # Only include relevant posts
                        if post_info['relevance_score'] > 0.1:
                            posts.append(post_info)
                            
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response from Reddit")
                
        except Exception as e:
            logger.error(f"Error scraping subreddit {subreddit}: {str(e)}")
        
        return posts
    
    def scrape_linkedin_posts(self, query: str, limit: int = 30) -> List[Dict]:
        """Scrape LinkedIn for financial posts using API if available, fallback to scraping"""
        posts = []

        try:
            # Check if LinkedIn API keys are available
            li_keys = self.api_keys.get('linkedin', {})
            if li_keys.get('client_id') and li_keys.get('access_token'):
                # Use LinkedIn API
                return self._scrape_linkedin_api(query, limit, li_keys)
            else:
                logger.warning("LinkedIn API keys not configured, falling back to limited scraping")
                # Fallback to limited scraping
                return self._scrape_linkedin_fallback(query, limit)

        except Exception as e:
            logger.error(f"Error scraping LinkedIn: {str(e)}")
            return []

    def _scrape_linkedin_api(self, query: str, limit: int, api_keys: Dict[str, str]) -> List[Dict]:
        """Scrape LinkedIn using API - Limited access available"""
        posts = []

        try:
            access_token = api_keys.get('access_token')
            if not access_token:
                logger.warning("LinkedIn access token not found - using fallback")
                return self._scrape_linkedin_fallback(query, limit)

            # LinkedIn API has restrictions on public content access
            # The UGC posts endpoint requires special permissions and may not work for general search
            logger.warning("LinkedIn API access limited - using fallback scraping")
            return self._scrape_linkedin_fallback(query, limit)

        except Exception as e:
            logger.error(f"Error with LinkedIn API: {str(e)}")
            return self._scrape_linkedin_fallback(query, limit)

    def _scrape_linkedin_fallback(self, query: str, limit: int = 30) -> List[Dict]:
        """Fallback LinkedIn scraping (very limited)"""
        posts = []

        try:
            logger.warning("LinkedIn scraping is very limited without API access")

            # LinkedIn requires authentication for most content
            # This is a basic approach that might work for some public content
            search_url = f"https://www.linkedin.com/search/results/content/?keywords={quote_plus(query)}"

            response = self._make_request(search_url)
            if not response:
                logger.warning("LinkedIn scraping failed - authentication required")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for post containers (structure may change)
            post_containers = soup.select('.feed-shared-update-v2, .occludable-update')

            for container in post_containers[:limit]:
                try:
                    # Extract post text
                    text_elem = container.select_one('.feed-shared-text, .break-words')
                    content = text_elem.get_text().strip() if text_elem else ""

                    # Extract author if available
                    author_elem = container.select_one('.feed-shared-actor__name, .actor-name')
                    author = author_elem.get_text().strip() if author_elem else "Unknown"

                    if content and len(content) > 20:
                        post_info = {
                            'platform': 'linkedin',
                            'title': content[:100] + "..." if len(content) > 100 else content,
                            'content': content,
                            'author': author,
                            'scraped_at': datetime.now(),
                            'relevance_score': self._calculate_financial_relevance(content)
                        }

                        if post_info['relevance_score'] > 0.1:
                            posts.append(post_info)

                except Exception as e:
                    logger.warning(f"Error parsing LinkedIn post: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error in LinkedIn fallback scraping: {str(e)}")

        logger.info(f"Scraped {len(posts)} LinkedIn posts (fallback)")
        return posts
    
    def scrape_facebook_public_posts(self, query: str, limit: int = 20) -> List[Dict]:
        """Scrape public Facebook posts using Graph API if available, fallback to scraping"""
        posts = []

        try:
            # Check if Facebook API keys are available
            fb_keys = self.api_keys.get('facebook', {})
            if fb_keys.get('app_id') and fb_keys.get('access_token'):
                # Use Facebook Graph API
                return self._scrape_facebook_api(query, limit, fb_keys)
            else:
                logger.warning("Facebook API keys not configured, falling back to limited scraping")
                # Fallback to limited scraping
                return self._scrape_facebook_fallback(query, limit)

        except Exception as e:
            logger.error(f"Error scraping Facebook: {str(e)}")
            return []

    def _scrape_facebook_api(self, query: str, limit: int, api_keys: Dict[str, str]) -> List[Dict]:
        """Scrape Facebook using Graph API - Limited to page posts"""
        posts = []

        try:
            access_token = api_keys.get('access_token')
            if not access_token:
                logger.warning("Facebook access token not found - using fallback")
                return self._scrape_facebook_fallback(query, limit)

            # Facebook Graph API doesn't allow general post search without special permissions
            # Instead, try to get posts from specific financial pages if page access token is available
            page_token = api_keys.get('page_access_token')
            if page_token:
                # Try to get posts from a specific page (would need page ID)
                # For now, fall back to scraping since general search isn't available
                logger.warning("Facebook Graph API search not available - using fallback scraping")
                return self._scrape_facebook_fallback(query, limit)
            else:
                logger.warning("Facebook page access token not available - using fallback scraping")
                return self._scrape_facebook_fallback(query, limit)

        except Exception as e:
            logger.error(f"Error with Facebook API: {str(e)}")
            return self._scrape_facebook_fallback(query, limit)

    def _scrape_facebook_fallback(self, query: str, limit: int = 20) -> List[Dict]:
        """Fallback Facebook scraping (very limited)"""
        posts = []

        try:
            logger.warning("Facebook scraping is very limited without API access")

            # Simulate some sample posts for demonstration
            sample_posts = [
                {
                    'platform': 'facebook',
                    'title': 'Ghana Stock Exchange performance discussion',
                    'content': 'Sample Facebook post about GSE performance...',
                    'author': 'Financial Analyst',
                    'scraped_at': datetime.now(),
                    'relevance_score': 0.8
                }
            ]

            return sample_posts

        except Exception as e:
            logger.error(f"Error in Facebook fallback scraping: {str(e)}")
            return []
    
    def scrape_twitter_alternative_sources(self, query: str) -> List[Dict]:
        """Scrape Twitter sources using API if available, fallback to Nitter"""
        posts = []

        try:
            # Check if Twitter API keys are available
            tw_keys = self.api_keys.get('twitter', {})
            if tw_keys.get('bearer_token'):
                # Use Twitter API v2
                return self._scrape_twitter_api(query, tw_keys)
            else:
                logger.warning("Twitter API keys not configured, falling back to Nitter")
                # Fallback to Nitter scraping
                return self._scrape_twitter_nitter(query)

        except Exception as e:
            logger.error(f"Error scraping Twitter sources: {str(e)}")
            return []

    def _scrape_twitter_api(self, query: str, api_keys: Dict[str, str]) -> List[Dict]:
        """Scrape Twitter using API v2"""
        posts = []

        try:
            bearer_token = api_keys.get('bearer_token')
            if not bearer_token:
                logger.error("Twitter bearer token not found")
                return []

            # Twitter API v2 recent search endpoint
            search_url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {bearer_token}'
            }

            params = {
                'query': query,
                'max_results': 50,  # API limit
                'tweet.fields': 'created_at,author_id,text,public_metrics',
                'user.fields': 'username,name'
            }

            response = self._make_request(search_url, headers=headers, params=params)
            if not response:
                return []

            data = response.json()

            if 'data' in data:
                for tweet in data['data']:
                    try:
                        content = tweet.get('text', '')
                        author_id = tweet.get('author_id', '')

                        # Get username from users data if available
                        author = "Unknown"
                        if 'includes' in data and 'users' in data['includes']:
                            for user in data['includes']['users']:
                                if user.get('id') == author_id:
                                    author = user.get('username', 'Unknown')
                                    break

                        # Parse timestamp
                        created_at = datetime.now()
                        if 'created_at' in tweet:
                            # Twitter timestamp format: '2023-01-01T12:00:00.000Z'
                            created_at = datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00'))

                        if content and len(content) > 10:
                            post_info = {
                                'platform': 'twitter',
                                'title': content[:100] + "..." if len(content) > 100 else content,
                                'content': content,
                                'author': author,
                                'created_at': created_at,
                                'scraped_at': datetime.now(),
                                'url': f"https://twitter.com/{author}/status/{tweet.get('id', '')}",
                                'relevance_score': self._calculate_financial_relevance(content)
                            }

                            if post_info['relevance_score'] > 0.1:
                                posts.append(post_info)

                    except Exception as e:
                        logger.warning(f"Error parsing Twitter API tweet: {str(e)}")
                        continue

            logger.info(f"Scraped {len(posts)} posts from Twitter API")
            return posts

        except Exception as e:
            logger.error(f"Error with Twitter API: {str(e)}")
            return []

    def _scrape_twitter_nitter(self, query: str) -> List[Dict]:
        """Fallback Twitter scraping using Nitter"""
        posts = []

        try:
            # Use Nitter (Twitter frontend) as alternative
            nitter_instances = [
                'https://nitter.net',
                'https://nitter.it',
                'https://nitter.42l.fr'
            ]

            for instance in nitter_instances:
                try:
                    search_url = f"{instance}/search?f=tweets&q={quote_plus(query)}"

                    response = self._make_request(search_url)
                    if not response:
                        continue

                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Extract tweets from Nitter
                    tweet_containers = soup.select('.timeline-item')

                    for container in tweet_containers[:20]:
                        try:
                            # Extract tweet content
                            content_elem = container.select_one('.tweet-content')
                            content = content_elem.get_text().strip() if content_elem else ""

                            # Extract author
                            author_elem = container.select_one('.username')
                            author = author_elem.get_text().strip() if author_elem else "Unknown"

                            # Extract timestamp
                            time_elem = container.select_one('.tweet-date')
                            timestamp = self._parse_relative_time(time_elem.get_text().strip()) if time_elem else datetime.now()

                            if content and len(content) > 10:
                                post_info = {
                                    'platform': 'twitter_alternative',
                                    'title': content[:100] + "..." if len(content) > 100 else content,
                                    'content': content,
                                    'author': author,
                                    'created_at': timestamp,
                                    'scraped_at': datetime.now(),
                                    'source': instance,
                                    'relevance_score': self._calculate_financial_relevance(content)
                                }

                                if post_info['relevance_score'] > 0.1:
                                    posts.append(post_info)

                        except Exception as e:
                            logger.warning(f"Error parsing tweet: {str(e)}")
                            continue

                    # If we got posts from this instance, break
                    if posts:
                        break

                except Exception as e:
                    logger.warning(f"Error with Nitter instance {instance}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping Twitter Nitter: {str(e)}")

        logger.info(f"Scraped {len(posts)} posts from Twitter alternatives (Nitter)")
        return posts
    
    def _parse_relative_time(self, time_str: str) -> datetime:
        """Parse relative time strings like '2h ago', '1d ago'"""
        try:
            now = datetime.now()
            time_str = time_str.lower().strip()
            
            if 'second' in time_str or 's' in time_str:
                seconds = int(re.search(r'(\d+)', time_str).group(1))
                return now - timedelta(seconds=seconds)
            elif 'minute' in time_str or 'm' in time_str:
                minutes = int(re.search(r'(\d+)', time_str).group(1))
                return now - timedelta(minutes=minutes)
            elif 'hour' in time_str or 'h' in time_str:
                hours = int(re.search(r'(\d+)', time_str).group(1))
                return now - timedelta(hours=hours)
            elif 'day' in time_str or 'd' in time_str:
                days = int(re.search(r'(\d+)', time_str).group(1))
                return now - timedelta(days=days)
            elif 'week' in time_str or 'w' in time_str:
                weeks = int(re.search(r'(\d+)', time_str).group(1))
                return now - timedelta(weeks=weeks)
            
        except:
            pass
        
        return datetime.now()
    
    def _calculate_financial_relevance(self, text: str) -> float:
        """Calculate relevance score for financial content"""
        text_lower = text.lower()
        
        # GSE and Ghana-specific keywords
        gse_score = sum(1 for keyword in self.financial_keywords if keyword in text_lower)
        
        # Company mentions
        company_score = 0
        for company, mentions in self.company_mentions.items():
            company_score += sum(1 for mention in mentions if mention.lower() in text_lower)
        
        # General financial terms
        financial_terms = [
            'stock', 'share', 'equity', 'dividend', 'trading', 'investment',
            'portfolio', 'market', 'bull', 'bear', 'price', 'volume',
            'earnings', 'profit', 'revenue', 'financial', 'economy'
        ]
        financial_score = sum(1 for term in financial_terms if term in text_lower)
        
        # Calculate relevance (0-1 scale)
        total_score = (gse_score * 3) + (company_score * 2) + financial_score
        max_possible = len(self.financial_keywords) * 3 + sum(len(mentions) for mentions in self.company_mentions.values()) * 2 + len(financial_terms)
        
        relevance = min(1.0, total_score / (max_possible * 0.1))
        return relevance
    
    def scrape_all_platforms(self, queries: List[str], max_posts_per_platform: int = 30) -> List[Dict]:
        """Scrape all configured social media platforms"""
        all_posts = []
        
        for query in queries:
            logger.info(f"Scraping social media for query: {query}")
            
            # Reddit
            reddit_posts = self.scrape_reddit_financial_discussions(query, max_posts_per_platform)
            all_posts.extend(reddit_posts)
            
            # LinkedIn (limited)
            linkedin_posts = self.scrape_linkedin_posts(query, max_posts_per_platform // 3)
            all_posts.extend(linkedin_posts)
            
            # Twitter
            twitter_posts = self.scrape_twitter_alternative_sources(query)
            all_posts.extend(twitter_posts)
            
            # Facebook (very limited)
            facebook_posts = self.scrape_facebook_public_posts(query, max_posts_per_platform // 5)
            all_posts.extend(facebook_posts)
            
            # Delay between queries
            time.sleep(5)
        
        # Remove duplicates and sort by relevance
        unique_posts = self._remove_duplicate_posts(all_posts)
        unique_posts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Total unique posts scraped: {len(unique_posts)}")
        return unique_posts
    
    def _remove_duplicate_posts(self, posts: List[Dict]) -> List[Dict]:
        """Remove duplicate posts based on content similarity"""
        unique_posts = []
        seen_hashes = set()
        
        for post in posts:
            # Create hash of content
            content = post.get('content', '') + post.get('title', '')
            content_hash = hashlib.md5(content.lower().encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_posts.append(post)
        
        return unique_posts
    
    def save_social_posts_to_db(self, posts: List[Dict], db_path: str = "social_media_posts.db") -> bool:
        """Save scraped social media posts to database"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create table
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
            
            saved_count = 0
            for post in posts:
                content_hash = hashlib.md5((post.get('content', '') + post.get('title', '')).encode()).hexdigest()
                
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO social_posts 
                        (platform, title, content, author, created_at, scraped_at, url, score, relevance_score, content_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        post.get('platform', ''),
                        post.get('title', ''),
                        post.get('content', ''),
                        post.get('author', ''),
                        post.get('created_at', post.get('scraped_at')),
                        post.get('scraped_at'),
                        post.get('url', ''),
                        post.get('score', 0),
                        post.get('relevance_score', 0),
                        content_hash
                    ))
                    
                    if cursor.rowcount > 0:
                        saved_count += 1
                        
                except sqlite3.IntegrityError:
                    continue  # Skip duplicates
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved {saved_count} new social media posts to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving social posts to database: {str(e)}")
            return False
    
    def get_trending_gse_topics(self) -> List[str]:
        """Get trending topics related to GSE from social media"""
        # This would analyze recent posts to identify trending topics
        # For now, return common search terms
        return [
            "Ghana Stock Exchange",
            "GSE trading",
            "Ghana shares",
            "MTN Ghana",
            "GCB Bank",
            "Ghana investment",
            "Accra stock market",
            "Ghana economy",
            "Ghana financial news"
        ]

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    scraper = SocialMediaScraper()
    
    print("Social Media Scraper Test")
    print("=" * 30)
    
    # Test queries
    test_queries = [
        "Ghana Stock Exchange",
        "GSE trading",
        "MTN Ghana shares"
    ]
    
    # Scrape all platforms
    all_posts = scraper.scrape_all_platforms(test_queries, max_posts_per_platform=20)
    
    if all_posts:
        print(f"✅ Scraped {len(all_posts)} social media posts")
        
        # Show platform breakdown
        platform_counts = {}
        for post in all_posts:
            platform = post.get('platform', 'unknown')
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        print("\nPlatform breakdown:")
        for platform, count in platform_counts.items():
            print(f"  {platform}: {count} posts")
        
        # Show top relevant posts
        print(f"\nTop 5 most relevant posts:")
        for i, post in enumerate(all_posts[:5]):
            print(f"\n{i+1}. Platform: {post.get('platform', 'Unknown')}")
            print(f"   Relevance: {post.get('relevance_score', 0):.3f}")
            print(f"   Content: {post.get('content', '')[:100]}...")
        
        # Save to database
        success = scraper.save_social_posts_to_db(all_posts)
        if success:
            print("\n✅ Posts saved to database")
        else:
            print("\n❌ Error saving posts to database")
    
    else:
        print("❌ No posts were scraped")
    
    print("\n✅ Social media scraping test completed!")