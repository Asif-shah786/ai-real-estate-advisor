"""
Simple configuration module for the AI Real Estate Assistant.

This module provides a clean way to access environment variables and configuration
throughout the project without dealing with SecretStr or complex validation.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Simple configuration class that loads values from environment variables."""

    def __init__(self):
        # API Keys
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
        self.langchain_endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "")

        # Model configurations
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-large"
        )
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.3"))

        # Database paths
        self.properties_file: str = os.getenv(
            "PROPERTIES_FILE", "datasets/run_ready_904.json"
        )
        self.legal_file: str = os.getenv(
            "LEGAL_FILE", "datasets/legal_uk_greater_manchester.jsonl"
        )
        self.use_existing_db: bool = (
            os.getenv("USE_EXISTING_DB", "true").lower() == "true"
        )
        self.db_name: str = os.getenv(
            "DB_NAME", "run_ready_904_legal_uk_greater_manchester"
        )
        # Memory and retrieval settings
        self.memory_token_limit: int = int(os.getenv("MEMORY_TOKEN_LIMIT", "2000"))
        self.retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "15"))
        self.retrieval_top_n: int = int(os.getenv("RETRIEVAL_TOP_N", "3"))

        # Evaluation settings
        self.evaluation_seed: int = int(os.getenv("EVALUATION_SEED", "42"))
        self.evaluation_questions: int = int(os.getenv("EVALUATION_QUESTIONS", "5"))

    def validate(self) -> bool:
        """Validate that required configuration is present."""
        if not self.openai_api_key:
            print("❌ OPENAI_API_KEY is required but not set")
            return False
        return True

    def get_openai_api_key(self) -> str:
        """Get OpenAI API key as a simple string."""
        return self.openai_api_key

    def get_langchain_api_key(self) -> str:
        """Get LangChain API key as a simple string."""
        return self.langchain_api_key

    def get_langchain_endpoint(self) -> str:
        """Get LangChain endpoint as a simple string."""
        return self.langchain_endpoint


# Create a global config instance
config = Config()


# Convenience functions for easy access
def get_openai_api_key() -> str:
    """Get OpenAI API key - simple function for easy access."""
    return config.get_openai_api_key()


def get_langchain_api_key() -> str:
    """Get LangChain API key - simple function for easy access."""
    return config.get_langchain_api_key()


def get_langchain_endpoint() -> str:
    """Get LangChain endpoint - simple function for easy access."""
    return config.get_langchain_endpoint()


def get_config() -> Config:
    """Get the global config instance."""
    return config


# Quick validation on import
if not config.validate():
    print("⚠️ Configuration validation failed. Some features may not work.")
