"""Configuration settings for the multi-agent system."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: str
    qdrant_host: Optional[str] = "localhost"
    qdrant_port: Optional[int] = 6333
    neo4j_uri: Optional[str] = "bolt://localhost:7687"
    neo4j_user: Optional[str] = "neo4j"
    neo4j_password: Optional[str] = ""
    
    # Model configurations
    default_llm_model: str = "gpt-4-turbo-preview"
    vision_model: str = "gpt-4-vision-preview"
    embedding_model: str = "text-embedding-3-large"
    
    # Tool configurations
    enable_web_search: bool = True
    enable_python_exec: bool = True
    max_tool_calls: int = 5
    
    # Agent configurations
    max_iterations: int = 10
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

