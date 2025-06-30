from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Dict, Any
import os

class Settings(BaseSettings):
    # Application settings
    app_name: str = "RAG Microservice"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # File processing settings
    max_document_size: int = 200 * 1024 * 1024  # 200MB
    temp_upload_dir: str = "/app/temp_uploads"
    data_dir: str = "/app/data"
    
    # Supported file extensions
    supported_extensions: Dict[str, List[str]] = {
        'images': ['.heic', '.jpg', '.jpeg', '.png', '.webp'],
        'documents': ['.doc', '.docx', '.odt'],
        'spreadsheets': ['.xls', '.xlsx', '.csv', '.ods'],
        'pdfs': ['.pdf'],
        'text': ['.txt', '.md', '.rtf']
    }
    
    # OpenAI settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    max_embedding_tokens: int = 8191
    max_completion_tokens: int = 8000
    
    # Qdrant settings
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_chunks"
    
    # Knowledge Graph settings
    graph_file: str = "/app/data/knowledge_graph.json"
    max_graph_depth: int = 3
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

settings = Settings()