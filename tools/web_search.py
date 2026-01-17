"""Web search tool for agents."""
from typing import List, Dict, Optional
import requests
from langchain.tools import Tool
from config import settings


class WebSearchTool:
    """Web search functionality."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool.
        
        Args:
            api_key: Optional API key for search service (e.g., SerpAPI, Tavily)
        """
        self.api_key = api_key
        self.enabled = settings.enable_web_search
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with 'title', 'url', 'snippet'
        """
        if not self.enabled:
            return []
        
        # Placeholder implementation
        # In production, integrate with SerpAPI, Tavily, or similar
        try:
            # Example with Tavily API (replace with actual implementation)
            if self.api_key:
                url = "https://api.tavily.com/search"
                response = requests.post(
                    url,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": num_results
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return [
                        {
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": r.get("content", "")
                        }
                        for r in data.get("results", [])
                    ]
        except Exception as e:
            print(f"Web search error: {e}")
        
        # Fallback: return empty results
        return []
    
    def get_tool(self) -> Tool:
        """Get LangChain Tool wrapper."""
        return Tool(
            name="web_search",
            description="Search the web for current information. Input should be a search query string.",
            func=self.search
        )

