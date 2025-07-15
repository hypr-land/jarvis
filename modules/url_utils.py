#!/usr/bin/env python3


import re
import requests
from urllib.parse import urlparse, parse_qs, quote_plus, unquote, urljoin
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class URLInfo:
    """Container for parsed URL information"""
    scheme: str
    netloc: str
    path: str
    query: Dict[str, List[str]]
    fragment: str
    full_url: str
    domain: str
    is_valid: bool


class URLUtils:
    """
    URL utilities for parsing, validation, and manipulation.
    
    Features:
    - URL parsing and validation
    - Domain extraction
    - Query parameter manipulation
    - URL normalization
    - Content type detection
    """
    
    # Common URL patterns
    URL_PATTERNS = {
        'http': r'^https?://',
        'ftp': r'^ftp://',
        'file': r'^file://',
        'mailto': r'^mailto:',
        'tel': r'^tel:',
        'data': r'^data:'
    }
    
    # Common file extensions
    FILE_EXTENSIONS = {
        'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp'],
        'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt'],
        'videos': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
        'audio': ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'],
        'archives': ['.zip', '.tar', '.gz', '.bz2', '.7z', '.rar'],
        'code': ['.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml']
    }
    
    def __init__(self, timeout: int = 10):
        """
        Initialize URL utilities.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def parse_url(self, url: str) -> URLInfo:
        """
        Parse URL and return structured information.
        
        Args:
            url: URL to parse
            
        Returns:
            URLInfo object with parsed URL components
        """
        try:
            parsed = urlparse(url)
            query_dict = parse_qs(parsed.query)
            
            # Extract domain from netloc
            domain = parsed.netloc.split(':')[0] if parsed.netloc else ''
            
            return URLInfo(
                scheme=parsed.scheme,
                netloc=parsed.netloc,
                path=parsed.path,
                query=query_dict,
                fragment=parsed.fragment,
                full_url=url,
                domain=domain,
                is_valid=self.is_valid_url(url)
            )
        except Exception:
            return URLInfo(
                scheme='', netloc='', path='', query={},
                fragment='', full_url=url, domain='', is_valid=False
            )
    
    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not url:
            return False
        
        # Check for basic URL pattern
        url_pattern = re.compile(
            r'^https?://'  # http or https
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
            r'localhost|'  # localhost
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP address
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(url))
    
    def normalize_url(self, url: str) -> str:
        """
        Normalize URL format.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        if not url:
            return url
        
        # Add scheme if missing
        if url.startswith('//'):
            return 'https:' + url
        elif not any(url.startswith(pattern) for pattern in self.URL_PATTERNS.values()):
            return 'https://' + url
        
        return url
    
    def extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Domain name
        """
        parsed = self.parse_url(url)
        return parsed.domain
    
    def get_file_extension(self, url: str) -> str:
        """
        Extract file extension from URL.
        
        Args:
            url: URL to extract extension from
            
        Returns:
            File extension (including dot)
        """
        path = urlparse(url).path
        return path[path.rfind('.'):] if '.' in path else ''
    
    def get_content_type(self, url: str) -> Optional[str]:
        """
        Get content type of URL (requires HTTP request).
        
        Args:
            url: URL to check
            
        Returns:
            Content type string or None
        """
        try:
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            return response.headers.get('content-type', '').split(';')[0]
        except Exception:
            return None
    
    def is_downloadable(self, url: str) -> bool:
        """
        Check if URL points to a downloadable file.
        
        Args:
            url: URL to check
            
        Returns:
            True if downloadable, False otherwise
        """
        content_type = self.get_content_type(url)
        if not content_type:
            return False
        
        # Check if content type indicates a file
        file_types = ['application/', 'image/', 'video/', 'audio/', 'text/']
        return any(content_type.startswith(ft) for ft in file_types)
    
    def categorize_url(self, url: str) -> str:
        """
        Categorize URL based on file extension or content type.
        
        Args:
            url: URL to categorize
            
        Returns:
            Category string
        """
        ext = self.get_file_extension(url).lower()
        
        for category, extensions in self.FILE_EXTENSIONS.items():
            if ext in extensions:
                return category
        
        # Check content type if no extension
        if not ext:
            content_type = self.get_content_type(url)
            if content_type:
                if content_type.startswith('image/'):
                    return 'images'
                elif content_type.startswith('video/'):
                    return 'videos'
                elif content_type.startswith('audio/'):
                    return 'audio'
                elif content_type.startswith('application/pdf'):
                    return 'documents'
        
        return 'unknown'
    
    def extract_query_params(self, url: str) -> Dict[str, List[str]]:
        """
        Extract query parameters from URL.
        
        Args:
            url: URL to extract parameters from
            
        Returns:
            Dictionary of parameter names to values
        """
        parsed = self.parse_url(url)
        return parsed.query
    
    def add_query_param(self, url: str, key: str, value: str) -> str:
        """
        Add query parameter to URL.
        
        Args:
            url: Base URL
            key: Parameter name
            value: Parameter value
            
        Returns:
            URL with added parameter
        """
        parsed = urlparse(url)
        query_dict = parse_qs(parsed.query)
        
        if key in query_dict:
            query_dict[key].append(value)
        else:
            query_dict[key] = [value]
        
        # Rebuild query string
        query_parts = []
        for k, v in query_dict.items():
            for val in v:
                query_parts.append(f"{quote_plus(k)}={quote_plus(val)}")
        
        new_query = '&'.join(query_parts)
        
        # Rebuild URL
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}{'#' + parsed.fragment if parsed.fragment else ''}"
    
    def remove_query_param(self, url: str, key: str) -> str:
        """
        Remove query parameter from URL.
        
        Args:
            url: URL to modify
            key: Parameter name to remove
            
        Returns:
            URL with parameter removed
        """
        parsed = urlparse(url)
        query_dict = parse_qs(parsed.query)
        
        if key in query_dict:
            del query_dict[key]
        
        # Rebuild query string
        query_parts = []
        for k, v in query_dict.items():
            for val in v:
                query_parts.append(f"{quote_plus(k)}={quote_plus(val)}")
        
        new_query = '&'.join(query_parts)
        
        # Rebuild URL
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{'?' + new_query if new_query else ''}{'#' + parsed.fragment if parsed.fragment else ''}"
    
    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs are from the same domain.
        
        Args:
            url1: First URL
            url2: Second URL
            
        Returns:
            True if same domain, False otherwise
        """
        domain1 = self.extract_domain(url1)
        domain2 = self.extract_domain(url2)
        return domain1 == domain2
    
    def get_redirect_url(self, url: str) -> Optional[str]:
        """
        Get final URL after following redirects.
        
        Args:
            url: URL to check
            
        Returns:
            Final URL after redirects or None
        """
        try:
            response = self.session.head(url, timeout=self.timeout, allow_redirects=True)
            return response.url
        except Exception:
            return None


# Example usage
if __name__ == "__main__":
    utils = URLUtils()
    
    # Test URL parsing
    test_url = "https://example.com/path?param=value#fragment"
    info = utils.parse_url(test_url)
    print(f"Domain: {info.domain}")
    print(f"Path: {info.path}")
    print(f"Query: {info.query}")
    
    # Test categorization
    image_url = "https://example.com/image.jpg"
    print(f"Category: {utils.categorize_url(image_url)}")
    
    # Test query parameter manipulation
    url = "https://example.com?param1=value1"
    new_url = utils.add_query_param(url, "param2", "value2")
    print(f"Modified URL: {new_url}") 