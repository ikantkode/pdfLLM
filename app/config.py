from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    openai_enabled: bool = Field(False, env="OPENAI_ENABLED")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field("gpt-4o-mini", env="OPENAI_CHAT_MODEL")

    # Ollama settings
    ollama_enabled: bool = Field(False, env="OLLAMA_ENABLED")
    ollama_host: str = Field("localhost", env="OLLAMA_HOST")
    ollama_port: int = Field(11434, env="OLLAMA_PORT")
    ollama_embedding_model: str = Field("bge-m3:latest", env="OLLAMA_EMBEDDING_MODEL")
    ollama_chat_model: str = Field("llama3.1:8b", env="OLLAMA_CHAT_MODEL")

    # Common model settings
    max_embedding_tokens: int = 8191
    max_completion_tokens: int = 4096

    # Qdrant settings
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # Dgraph settings
    dgraph_host: str = "dgraph"
    dgraph_port: int = 9080
    dgraph_token: str = Field("", env="DGRAPH_TOKEN")

    # Knowledge Graph settings
    max_graph_depth: int = 3

    # Settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @model_validator(mode='after')
    def log_settings(self):
        logger.info(f"OPENAI_ENABLED: {self.openai_enabled}")
        logger.info(f"OLLAMA_ENABLED: {self.ollama_enabled}")
        return self

settings = Settings()