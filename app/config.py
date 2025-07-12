from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Dict

class Settings(BaseSettings):
    # Application settings
    app_name: str = "PDFLLM RAG Microservice"
    app_version: str = "1.0.0"
    debug: bool = False

    # File processing settings
    max_document_size: int = 50 * 1024 * 1024  # 50MB
    temp_upload_dir: str = "./data/temp_uploads"
    data_dir: str = "./data"

    # Supported file extensions
    supported_extensions: Dict[str, List[str]] = {
        "pdfs": [".pdf"],
        "images": [".jpg", ".jpeg", ".png", ".heic", ".webp"],
        "documents": [".doc", ".docx", ".odt"],
        "spreadsheets": [".xls", ".xlsx", ".csv", ".ods"],
        "text": [".txt", ".md", ".rtf"]
    }

    # OpenAI settings
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4-turbo"
    max_embedding_tokens: int = 8191
    max_completion_tokens: int = 4096

    # Qdrant settings
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Knowledge Graph settings
    graph_file: str = "./data/knowledge_graph.json"
    max_graph_depth: int = 3

    # Settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance
settings = Settings()