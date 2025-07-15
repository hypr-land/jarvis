#!/usr/bin/env python3


import requests
import re
from urllib.parse import urlparse, parse_qs, quote_plus, unquote, urljoin
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import requests
import re
from urllib.robotparser import RobotFileParser
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search result data"""
    title: str
    url: str
    content: str
    summary: str
    relevance_score: float = 0.0


class SearchEngine:
    """
    Web search engine with DuckDuckGo integration and content summarization.
    
    Features:
    - DuckDuckGo search integration
    - Web page content scraping
    - Content summarization with AI
    - Configurable search parameters
    - Error handling and retry logic
    """
    
    def __init__(self, 
                 user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                 timeout: int = 10,
                 max_content_length: int = 8192):
        """
        Initialize search engine.
        
        Args:
            user_agent: User agent string for requests
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to process
        """
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
        self.robots_cache: Dict[str, Tuple[RobotFileParser, float]] = {}
        logger.info(f"SearchEngine initialized with user agent: {user_agent}")
        
    def _check_robots_txt(self, url: str) -> bool:
        """
        Check if URL is allowed by the site's robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if crawling is allowed, False otherwise
        """
        try:
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            robots_url = urljoin(base_url, "/robots.txt")
            
            logger.info(f"Checking robots.txt for: {base_url}")
            
            # Check cache first
            if base_url in self.robots_cache:
                rp, timestamp = self.robots_cache[base_url]
                # Cache robots.txt for 24 hours
                if time.time() - timestamp < 86400:
                    allowed = rp.can_fetch(self.user_agent, url)
                    logger.info(f"Using cached robots.txt - Access {'allowed' if allowed else 'denied'} for {url}")
                    return allowed
            
            # Fetch and parse robots.txt
            rp = RobotFileParser()
            rp.set_url(robots_url)
            try:
                logger.info(f"Fetching robots.txt from: {robots_url}")
                rp.read()
                self.robots_cache[base_url] = (rp, time.time())
                allowed = rp.can_fetch(self.user_agent, url)
                logger.info(f"Fresh robots.txt check - Access {'allowed' if allowed else 'denied'} for {url}")
                return allowed
            except Exception as e:
                logger.warning(f"Error reading robots.txt for {base_url}: {e}")
                # If we can't read robots.txt, assume crawling is allowed
                return True
            
        except Exception as e:
            logger.error(f"Error checking robots.txt: {e}")
            # If there's an error, assume crawling is allowed
            return True
    
    def search(self, 
               query: str, 
               max_results: int = 3, 
               summarize: bool = True,
               groq_client: Optional[Any] = None) -> List[SearchResult]:
        """
        Perform web search and return results.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            summarize: Whether to summarize content
            groq_client: Optional Groq client for summarization
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Perform DuckDuckGo search
            results = self._search_duckduckgo(query, max_results)
            
            # Process and summarize results
            processed_results = []
            for result in results:
                processed_result = self._process_result(result, summarize, groq_client)
                if processed_result:
                    processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Perform DuckDuckGo search and extract results."""
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            
            for result in soup.select("a.result__a")[:max_results]:
                title = result.get_text().strip()
                raw_link = result.get("href", "")
                link = self._extract_real_url(raw_link)
                link = self._normalize_url(link)
                
                if title and link:
                    results.append({
                        "title": title,
                        "url": link
                    })
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []
    
    def _process_result(self, 
                       result: Dict[str, str], 
                       summarize: bool,
                       groq_client: Optional[Any]) -> Optional[SearchResult]:
        """Process a search result and optionally summarize content."""
        try:
            # Scrape content
            content = self._scrape_content(result["url"])
            if not content:
                return None
            
            # Truncate content if needed
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
            
            # Summarize if requested and client available
            summary = ""
            if summarize and groq_client:
                summary = self._summarize_content(content, groq_client)
            else:
                summary = content[:200] + "..." if len(content) > 200 else content
            
            return SearchResult(
                title=result["title"],
                url=result["url"],
                content=content,
                summary=summary
            )
            
        except Exception as e:
            print(f"Error processing result: {e}")
            return None
    
    def _scrape_content(self, url: str) -> str:
        """Scrape content from a web page."""
        try:
            # Check robots.txt first
            if not self._check_robots_txt(url):
                logger.warning(f"URL not allowed by robots.txt: {url}")
                return ""
            
            logger.info(f"Scraping content from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "iframe"]):
                tag.decompose()
            
            # Extract text
            text = soup.get_text(separator="\n")
            
            # Clean up whitespace
            text = re.sub(r"\n{2,}", "\n", text)
            text = re.sub(r"\s+", " ", text)
            text = text.strip()
            
            logger.info(f"Successfully scraped {len(text)} chars from {url}")
            return text
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return ""
    
    def _summarize_content(self, content: str, groq_client: Any) -> str:
        """Summarize content using Groq API."""
        try:
            # Create summarization prompt
            prompt = f"""Summarize the following content in 2-3 sentences, focusing on the key points:

{content[:4000]}  # Limit content for API

Provide a concise summary that captures the main information."""
            
            # Call Groq API
            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-mini-7b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Summarization error: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL format."""
        if url.startswith("//"):
            return "https:" + url
        elif url.startswith("http://") or url.startswith("https://"):
            return url
        else:
            return "https://" + url
    
    def _extract_real_url(self, redirect_url: Optional[str]) -> str:
        """Extract real URL from DuckDuckGo redirect."""
        try:
            if not redirect_url:
                return ""
                
            parsed = urlparse(redirect_url)
            qs = parse_qs(parsed.query)
            if 'uddg' in qs:
                real_url = qs['uddg'][0]
                return unquote(real_url)
            else:
                return redirect_url
        except Exception:
            return redirect_url or ""
    
    def search_simple(self, query: str, max_results: int = 3) -> str:
        """
        Simple search that returns formatted string results.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Formatted string with search results
        """
        results = self.search(query, max_results, summarize=False)
        
        if not results:
            return f"No search results found for: {query}"
        
        formatted_results = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted_results += f"{i}. {result.title}\n"
            formatted_results += f"   URL: {result.url}\n"
            formatted_results += f"   Summary: {result.summary}\n\n"
        
        return formatted_results.strip()


class SearchConfig:
    """Configuration for search functionality."""
    
    def __init__(self,
                 enabled: bool = True,
                 max_results: int = 3,
                 timeout: int = 10,
                 user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                 max_content_length: int = 8192):
        self.enabled = enabled
        self.max_results = max_results
        self.timeout = timeout
        self.user_agent = user_agent
        self.max_content_length = max_content_length


# Example usage
if __name__ == "__main__":
    # Create search engine
    search_engine = SearchEngine()
    
    # Perform search
    query = "Python programming tutorials"
    results = search_engine.search_simple(query, max_results=2)
    
    print(results) 