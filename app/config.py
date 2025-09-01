
import os
from typing import Dict, Optional
from urllib.parse import urlparse
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Application settings
    app_name: str = "PDFLLM RAG Microservice"
    app_version: str = "1.0.0"
    debug: bool = False

    # File processing settings
    max_document_size: int = 52428800  # 50MB
    temp_upload_dir: str = "./data/temp_uploads"
    data_dir: str = "./data"
    supported_extensions: Dict[str, list] = {
        "images": [".jpg", ".jpeg", ".png", ".heic", ".webp"],
        "documents": [".doc", ".docx", ".odt"],
        "spreadsheets": [".xls", ".xlsx", ".csv", ".ods"],
        "text": [".txt", ".md", ".rtf"],
        "pdfs": [".pdf"]
    }
    max_embedding_tokens: int = 8191
    max_completion_tokens: int = 4096

    # OpenAI settings
    openai_enabled: bool = True
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"  # Default OpenAI endpoint

    # Qdrant settings
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "pdfllm_collection"

    # Dgraph settings
    dgraph_host: str = "dgraph"
    dgraph_port: int = 9080
    dgraph_token: Optional[str] = None

    # PostgreSQL settings
    postgres_host: str = "postgres"
    postgres_port: int = 9011
    postgres_db: str = "ragdb"
    postgres_user: str = "raguser"
    postgres_password: str = "ragpassword"

    # Celery settings
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_settings(self):
        if self.openai_enabled and not self.openai_api_key:
            logger.error("OPENAI_API_KEY is required when OPENAI_ENABLED is true")
            raise ValueError("OPENAI_API_KEY is required when OPENAI_ENABLED is true")
        
        # Validate OpenAI base URL
        try:
            result = urlparse(self.openai_base_url)
            if not all([result.scheme, result.netloc]):
                logger.error(f"Invalid OPENAI_BASE_URL: {self.openai_base_url}. Must be a valid URL (e.g., https://api.openai.com/v1)")
                raise ValueError(f"Invalid OPENAI_BASE_URL: {self.openai_base_url}")
        except Exception as e:
            logger.error(f"Failed to parse OPENAI_BASE_URL: {str(e)}")
            raise ValueError(f"Invalid OPENAI_BASE_URL: {str(e)}")

settings = Settings()
settings.validate_settings()
logger.info("Settings initialized successfully")