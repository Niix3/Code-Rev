"""Configuration settings for the multi-agent system."""
from enum import Enum
from typing import Optional

from pydantic_settings import BaseSettings


class TesterMode(str, Enum):
    """How the tester agent validates implementation."""

    BENCHMARKING = "benchmarking"
    TEST_GENERATION = "test_generation"


class Settings(BaseSettings):
    """Application settings."""

    # API Keys
    openai_api_key: str
    openai_base_url: str = "https://api.aitunnel.ru/v1/"

    # Model configurations
    default_llm_model: str = "deepseek-r1"

    # Tool configurations
    enable_web_search: bool = True
    enable_python_exec: bool = True
    max_tool_calls: int = 5

    # Agent configurations
    max_iterations: int = 10
    temperature: float = 0.7
    workspace_path: str = "/workspace"

    # OpenHands integration
    openhands_api_key: Optional[str] = None
    # OpenHands → LiteLLM: нужен префикс провайдера (openai/… для совместимого с OpenAI API).
    openhands_model: str = "openai/deepseek-r1"
    openhands_llm_base_url: Optional[str] = None
    openhands_timeout_seconds: int = 180

    # Tester configuration
    tester_mode: TesterMode = TesterMode.TEST_GENERATION
    tester_command: str = "pytest -q"
    tester_timeout_seconds: int = 180

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


settings = Settings()
