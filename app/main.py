import os
import uuid
import json
import logging
import base64
import datetime
import hashlib
import re
from config import settings
from pydgraph import RetriableError
from typing import List, Dict, Optional, Any
from io import BytesIO
from pathlib import Path
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    status,
    Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.encoders import jsonable_encoder
import tiktoken
import pydgraph
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import requests
from pydantic import BaseModel
from utils.qdrant_handler import QdrantHandler
from utils.text_processor import TextProcessor
from utils.ocr_processor import OCRProcessor
from converters import (
    image_converter,
    doc_converter,
    excel_converter,
    txt_converter
)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Configure logging
Path(settings.data_dir).joinpath("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(settings.data_dir) / "logs" / "rag_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A Retrieval-Augmented Generation (RAG) microservice",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key")

async def validate_api_key(api_key: str = Depends(api_key_header)):
    if settings.openai_enabled and api_key != settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Check provider configuration
def validate_provider_settings():
    if settings.openai_enabled and settings.ollama_enabled:
        logger.error("Both OpenAI and Ollama are enabled. Only one provider can be active.")
        raise HTTPException(
            status_code=400,
            detail="Invalid configuration: Both OPENAI_ENABLED and OLLAMA_ENABLED are set to true. Please enable only one provider."
        )
    if not settings.openai_enabled and not settings.ollama_enabled:
        logger.error("No provider enabled. Either OPENAI_ENABLED or OLLAMA_ENABLED must be set to true.")
        raise HTTPException(
            status_code=400,
            detail="Invalid configuration: Neither OPENAI_ENABLED nor OLLAMA_ENABLED is set to true."
        )

# Initialize OCR processor
ocr_processor = OCRProcessor()

# Initialize Qdrant with retry logic
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    reraise=True
)
def initialize_qdrant_handler():
    try:
        qdrant_handler = QdrantHandler()
        logger.info("Qdrant connection established")
        return qdrant_handler
    except Exception as e:
        logger.error(f"Qdrant connection attempt failed: {str(e)}")
        raise

try:
    qdrant_handler = initialize_qdrant_handler()
except Exception as e:
    logger.error(f"Qdrant connection failed after retries: {str(e)}")
    raise

try:
    text_processor = TextProcessor()
    logger.info("TextProcessor initialized successfully")
except Exception as e:
    logger.error(f"TextProcessor initialization failed: {str(e)}")
    raise

# Create necessary directories
Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)

# Models
class FileMetadata(BaseModel):
    file_id: str
    filename: str
    file_type: str
    upload_date: str
    content: str
    markdown_content: str
    user_id: str
    size: int
    checksum: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class ChatSession(BaseModel):
    chat_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    document_ids: List[str]

class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    parent_section: str
    chunk_index: int
    content: str
    entities: List[str]
    relationships: List[Dict[str, str]]
    score: float

# Knowledge Graph for advanced indexing
class KnowledgeGraph:
    def __init__(self):
        self.client = pydgraph.DgraphClient(
            pydgraph.DgraphClientStub(f"{settings.dgraph_host}:{settings.dgraph_port}")
        )
        self._initialize_schema()
        self.entity_index = {}

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
        retry=retry_if_exception_type(RetriableError)
    )
    def _initialize_schema(self):
        """Initialize Dgraph schema for knowledge graph with retry logic"""
        schema = """
            type Entity {
                name
                type
                appears_in
            }
            type Document {
                document_id
            }
            name: string @index(exact) .
            type: string @index(exact) .
            document_id: string @index(exact) .
            appears_in: [uid] @reverse .
            relationship: [uid] @reverse .
            predicate: string .
            weight: float .
        """
        try:
            op = pydgraph.Operation(schema=schema)
            self.client.alter(op)
            logger.info("Dgraph schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Dgraph schema: {str(e)}")
            raise

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship: str,
        weight: float = 1.0
    ) -> None:
        """Add a relationship between entities or entity and document"""
        source = re.sub(r'[\n\r\t]', ' ', source).strip()
        target = re.sub(r'[\n\r\t]', ' ', target).strip()
        source = re.sub(r'\s+', ' ', source)
        target = re.sub(r'\s+', ' ', target)
        
        if (not source or not target or len(source) > 255 or len(target) > 255 or
            re.search(r'\b(Date|Signature)\b', source, re.IGNORECASE) or
            re.search(r'\b(Date|Signature)\b', target, re.IGNORECASE)):
            logger.warning(f"Skipping invalid relationship: source='{source}', target='{target}', relationship='{relationship}'")
            return

        logger.debug(f"Adding relationship: {source} -> {relationship} -> {target}")
        txn = self.client.txn()
        try:
            query = f"""
                query {{
                    source(func: eq(name, "{source}")) {{
                        uid
                        name
                        type
                    }}
                }}
            """
            res = txn.query(query)
            source_node = json.loads(res.json).get("source", [])

            if not source_node:
                source_node = {
                    "uid": "_:source",
                    "name": source,
                    "type": "entity"
                }
                mutation = {
                    "uid": "_:source",
                    "name": source,
                    "type": "entity"
                }
            else:
                source_node = source_node[0]
                mutation = {"uid": source_node["uid"]}

            query = f"""
                query {{
                    target(func: eq(document_id, "{target}")) {{
                        uid
                        document_id
                    }}
                    entity(func: eq(name, "{target}")) {{
                        uid
                        name
                        type
                    }}
                }}
            """
            res = txn.query(query)
            target_data = json.loads(res.json)
            target_node = target_data.get("target", []) or target_data.get("entity", [])

            if not target_node:
                target_node = {
                    "uid": "_:target",
                    "name": target,
                    "type": "entity"
                }
                mutation["relationship"] = [{
                    "uid": "_:target",
                    "name": target,
                    "type": "entity",
                    "predicate": relationship,
                    "weight": weight
                }]
            else:
                target_node = target_node[0]
                if "document_id" in target_node:
                    mutation["appears_in"] = [{
                        "uid": target_node["uid"],
                        "document_id": target,
                        "predicate": relationship,
                        "weight": weight
                    }]
                else:
                    mutation["relationship"] = [{
                        "uid": target_node["uid"],
                        "predicate": relationship,
                        "weight": weight
                    }]

            txn.mutate(set_obj=mutation)
            txn.commit()
            logger.debug(f"Added relationship: {source} -> {relationship} -> {target}")
        except Exception as e:
            logger.error(f"Failed to add relationship: {source} -> {relationship} -> {target}, error: {str(e)}")
        finally:
            txn.discard()

    def find_related_entities(
        self,
        entity: str,
        depth: int = settings.max_graph_depth
    ) -> List[str]:
        """Find related entities within specified depth"""
        entity = re.sub(r'[\n\r\t]', ' ', entity).strip()
        entity = re.sub(r'\s+', ' ', entity)
        if not entity or len(entity) > 255 or re.search(r'\b(Date|Signature)\b', entity, re.IGNORECASE):
            logger.warning(f"Invalid entity for search: {entity}")
            return []

        txn = self.client.txn(read_only=True)
        try:
            query = f"""
                query {{
                    var(func: eq(name, "{entity}")) {{
                        e as uid
                    }}
                    related(func: uid(e)) @recurse(depth: {depth}, loop: false) {{
                        name
                        ~relationship
                        ~appears_in
                    }}
                }}
            """
            res = txn.query(query)
            data = json.loads(res.json).get("related", [])
            entities = [item["name"] for item in data if item.get("name") and item.get("type") == "entity"]
            return entities
        except Exception as e:
            logger.error(f"Failed to find related entities: {str(e)}")
            return []
        finally:
            txn.discard()

    def save(self) -> None:
        """No-op: Dgraph handles persistence natively"""
        logger.info("Dgraph handles persistence, no explicit save needed")

    def load(self) -> None:
        """No-op: Dgraph handles persistence natively"""
        logger.info("Dgraph handles persistence, no explicit load needed")

knowledge_graph = KnowledgeGraph()

# State management
class StateManager:
    def __init__(self):
        self.state_file = Path(settings.data_dir) / "state.json"
        self.file_metadata: List[Dict[str, Any]] = []
        self.chat_sessions: Dict[str, Dict[str, Any]] = {}
        self.load()

    def save(self) -> None:
        """Save application state"""
        state = {
            "file_metadata": self.file_metadata,
            "chat_sessions": self.chat_sessions
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(jsonable_encoder(state), f)
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def load(self) -> None:
        """Load application state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.file_metadata = state.get("file_metadata", [])
                    self.chat_sessions = state.get("chat_sessions", {})
                logger.info("State loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load state: {str(e)}")

state_manager = StateManager()

# Helper functions
def get_file_converter(file_ext: str):
    """Get the appropriate converter for a file extension"""
    if file_ext in settings.supported_extensions['images']:
        return image_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['documents']:
        return doc_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['spreadsheets']:
        return excel_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['text']:
        return txt_converter.convert_to_markdown
    return None

def split_large_text(text: str, max_tokens: int = settings.max_embedding_tokens) -> List[str]:
    """Split text into chunks, preserving table structures and semantic units"""
    table_pattern = r'(\|.*?\|\n(?:\|[-: ]+\|\n)+.*?)(?=\n\n|\Z)'
    tables = re.findall(table_pattern, text, re.DOTALL)
    non_table_text = re.sub(table_pattern, 'TABLE_PLACEHOLDER', text)

    chunks = []
    current_chunk = ""
    token_count = 0
    placeholder_count = 0

    doc = text_processor.nlp(non_table_text)
    for sent in doc.sents:
        sent_text = sent.text
        if 'TABLE_PLACEHOLDER' in sent_text:
            if placeholder_count < len(tables):
                sent_text = sent_text.replace('TABLE_PLACEHOLDER', tables[placeholder_count])
                placeholder_count += 1
        sent_tokens = len(tokenizer.encode(sent_text))

        if token_count + sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                token_count = 0
            if sent_tokens > max_tokens:
                sub_chunks = [sent_text[i:i + max_tokens] for i in range(0, len(sent_text), max_tokens)]
                chunks.extend(sub_chunks)
                continue

        current_chunk += sent_text + "\n"
        token_count += sent_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    while placeholder_count < len(tables):
        table_tokens = len(tokenizer.encode(tables[placeholder_count]))
        if table_tokens <= max_tokens:
            chunks.append(tables[placeholder_count])
        else:
            rows = tables[placeholder_count].split('\n')
            sub_chunk = ""
            sub_tokens = 0
            for row in rows:
                row_tokens = len(tokenizer.encode(row))
                if sub_tokens + row_tokens > max_tokens:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                        sub_chunk = ""
                        sub_tokens = 0
                    sub_chunk += row + "\n"
                    sub_tokens += row_tokens
                else:
                    sub_chunk += row + "\n"
                    sub_tokens += row_tokens
            if sub_chunk:
                chunks.append(sub_chunk.strip())
        placeholder_count += 1

    logger.info(f"Split text into {len(chunks)} chunks with max {max_tokens} tokens")
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings with batch processing and retry logic"""
    validate_provider_settings()
    try:
        if settings.openai_enabled:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.embeddings.create(
                input=texts,
                model=settings.openai_embedding_model,
                dimensions=1024  # Truncate to 1024 dimensions
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"OpenAI embeddings generated with {len(embeddings[0])} dimensions")
            return embeddings
        else:
            embeddings = []
            for text in texts:
                response = requests.post(
                    f"http://{settings.ollama_host}:{settings.ollama_port}/api/embeddings",
                    json={
                        "model": settings.ollama_embedding_model,
                        "prompt": text  # Send single string
                    }
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data["embedding"])
            logger.info(f"Ollama embeddings generated with {len(embeddings[0])} dimensions")
            return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def preprocess_ocr_text(text: str) -> str:
    """Use LLM to clean and reconstruct OCR-extracted text into coherent sentences"""
    validate_provider_settings()
    chunks = split_large_text(text, max_tokens=settings.max_embedding_tokens // 2)
    cleaned_chunks = []

    if settings.openai_enabled:
        client = OpenAI(api_key=settings.openai_api_key)
    else:
        client = None  # We'll use requests for Ollama

    for chunk in chunks:
        try:
            prompt = (
                f"Input Text:\n{chunk}\n\n"
                "You are a document cleaner. Remove all OCR artifacts first, then re-format the remainder.\n\n"
                "Step-0 (MANDATORY): Collapse every space inside a token (words, names, numbers, currency, dates). "
                "Example: ‘1 , 0 0 0 , 0 0 0’ → ‘1,000,000’, ‘A d d i t i o n a l’ → ‘Additional’. "
                "Do not remove spaces that separate distinct tokens.\n\n"
                "Steps 1-11:\n"
                "1. Reconstruct the text into clear, grammatical, logically structured content.\n"
                "2. Standardize numbers and currency (single ‘$’, commas for thousands, two decimals).\n"
                "3. Rejoin broken names/words that remain after Step-0.\n"
                "4. Separate metadata labels (Date, Signature, etc.) from the preceding name/value.\n"
                "5. Preserve markdown headings, lists, and tables; ensure tables are well-aligned.\n"
                "6. If payroll data (employee, role, hours, rate, gross/net pay) is present:\n"
                "   - Render it in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay.\n"
                "   - Validate: Gross Pay ≈ Hours × Rate, Net Pay < Gross Pay. Note any assumptions.\n"
                "7. If not payroll-related, simply return clean markdown prose.\n"
                "8. Remove gibberish or noise; keep all meaningful data.\n"
                "9. Comment assumptions with <!-- ... -->.\n"
                "10. Return only the cleaned markdown.\n"
                "11. Stay under the token budget.\n"
                    )

            if settings.openai_enabled:
                response = client.chat.completions.create(
                    model=settings.openai_chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert at cleaning and reconstructing noisy OCR-extracted text. "
                                "Correct OCR errors, including spaces between characters in words, names, or numbers. "
                                "Separate names from metadata like 'Date' or 'Signature' (e.g., 'Aslam Baig \nDate' → 'Aslam Baig'). "
                                "Format numbers and names properly, ensuring single '$' for currency, and structure the output in clear, coherent markdown. "
                                "Preserve meaningful information and structural elements (e.g., lists, tables). "
                                "For payroll data, format details in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay. "
                                "Validate numerical consistency: Gross Pay = Hours × Rate, Net Pay < Gross Pay. Use context-provided rates (e.g., $94.43 per hour) if available. "
                                "Note any assumptions made due to ambiguous text in markdown comments."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=settings.max_completion_tokens,
                    temperature=0.3
                )
                cleaned_chunk = response.choices[0].message.content.strip()
            else:
                logger.debug(f"Sending OCR preprocessing request to Ollama for chunk: {chunk[:100]}...")
                response = requests.post(
                    f"http://{settings.ollama_host}:{settings.ollama_port}/v1/chat/completions",
                    json={
                        "model": settings.ollama_chat_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert at cleaning and reconstructing noisy OCR-extracted text. "
                                    "Correct OCR errors, including spaces between characters in words, names, or numbers. "
                                    "Separate names from metadata like 'Date' or 'Signature' (e.g., 'Aslam Baig \nDate' → 'Aslam Baig'). "
                                    "Format numbers and names properly, ensuring single '$' for currency, and structure the output in clear, coherent markdown. "
                                    "Preserve meaningful information and structural elements (e.g., lists, tables). "
                                    "For payroll data, format details in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay. "
                                    "Validate numerical consistency: Gross Pay = Hours × Rate, Net Pay < Gross Pay. Use context-provided rates (e.g., $94.43 per hour) if available. "
                                    "Note any assumptions made due to ambiguous text in markdown comments."
                                    "Return only the cleaned text in markdown format, without wrapping in JSON or markdown code blocks."
                                )
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": settings.max_completion_tokens,
                        "temperature": 0.3
                    }
                )
                response.raise_for_status()
                logger.debug(f"Ollama API response status: {response.status_code}")
                cleaned_chunk = response.json()["choices"][0]["message"]["content"].strip()
                logger.debug(f"Cleaned OCR response: {cleaned_chunk[:200]}...")

            cleaned_chunks.append(cleaned_chunk)
            logger.debug(f"Preprocessed chunk: {cleaned_chunk[:200]}...")
        except Exception as e:
            logger.error(f"Failed to preprocess OCR chunk: {str(e)}")
            raise

    cleaned_text = "\n\n".join(cleaned_chunks)
    logger.debug(f"Preprocessed OCR text: {cleaned_text[:500]}...")
    return cleaned_text

def clean_response(text: str) -> str:
    """Final cleanup of LLM output with aggressive number and formatting fixes"""
    logger.debug(f"Raw response before cleaning: {text[:500]}...")

    # Phase 1: Fix number ranges and currency (most critical first)
    text = re.sub(r'(\d{1,3}(?:,\d{3})*)\s*\n\s*([a-zA-Z]+)\s*\n\s*(\d{1,3}(?:,\d{3})*)', r'\1 \2 \3', text)  # 1\nmillion\n3 → 1 million 3
    text = re.sub(r'(\d+)\s*\n\s*([,.])\s*\n\s*(\d+)', r'\1\2\3', text)  # 1\n,\n000 → 1,000
    text = re.sub(r'(\d+)\s*-\s*\n\s*(\d+)', r'\1-\2', text)  # 1-\n000 → 1-000
    text = re.sub(r'\$\s*\n\s*(\d+)', r'$\1', text)  # $\n100 → $100

    # Phase 2: Fix spaced-out words and numbers
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)
    text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)
    text = re.sub(r'(\d)\s+([kKmMbB])\b', r'\1\2', text)

    # Phase 3: Clean up remaining artifacts
    text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)  # Newlines between words
    text = re.sub(r'(?<=\d)\n(?=\d)', '', text)   # Newlines between digits
    text = re.sub(r'(?<=\$)\n(?=\d)', '', text)   # Newlines after $
    text = re.sub(r'[ \t]{2,}', ' ', text)        # Multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text)        # Multiple newlines

    # Phase 4: Final formatting pass
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    text = '\n\n'.join(
        f"{p[0].upper()}{p[1:]}" if p and p[0].islower() else p
        for p in paragraphs
    )

    logger.debug(f"Cleaned response: {text[:500]}...")
    return text.strip()

def format_search_results(results: List) -> List[SearchResult]:
    """Format search results for response"""
    file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}
    return [
        SearchResult(
            chunk_id=str(r.id),
            document_id=r.payload.get("document_id", "N/A"),
            filename=file_map.get(r.payload.get("document_id", "N/A"), "Unknown"),
            parent_section=r.payload.get("parent_section", "N/A"),
            chunk_index=r.payload.get("chunk_index", 0),
            content=r.payload.get("content", "N/A"),
            entities=r.payload.get("entities", []),
            relationships=r.payload.get("relationships", []),
            score=r.score
        )
        for r in results
    ]

def rank_results(entity_results: List, vector_results: List, limit: int) -> List:
    """Combine and rank results from different retrieval methods"""
    seen = set()
    combined = []

    for result in entity_results + vector_results:
        result_id = result.id if hasattr(result, 'id') else result.get('chunk_id')
        if result_id and result_id not in seen:
            seen.add(result_id)
            score = (result.score if hasattr(result, 'score') else result.get('score', 0.0))
            score += 0.1 * len(result.payload.get('entities', []))
            score += 0.2 * len(result.payload.get('relationships', []))
            result.score = score
            combined.append(result)

    return sorted(
        combined,
        key=lambda x: x.score if hasattr(x, 'score') else x.get('score', 0.0),
        reverse=True
    )[:limit]

def enforce_response_quality(text: str) -> str:
    """Enforce consistent formatting and remove all artifacts"""
    # Remove newlines within words/numbers
    text = re.sub(r'(?<=\w)\n(?=\w)', '', text)
    text = re.sub(r'(?<=\d)\n(?=\d)', '', text)
    
    # Fix spaced-out characters
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)
    
    # Format numbers/currency
    text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)
    text = re.sub(r'\$\s+(\d)', r'$\1', text)
    text = re.sub(r'(\d)\s+(\d{3})\s+(\d{3})', r'\1,\2,\3', text)
    
    # Ensure proper paragraph structure
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    text = '\n\n'.join(
        f"{p[0].upper()}{p[1:]}" if p else "" 
        for p in paragraphs
    )
    
    return text.strip()

def clean_query(query: str) -> str:
    """Clean user query to handle OCR-like noise and payroll-specific formatting"""
    query = re.sub(r'[\n\r\t]', ' ', query).strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'([^\w\s])\1+', r'\1', query)
    query = re.sub(r'\b(\w+)\s+\1\b', r'\1', query, flags=re.IGNORECASE)
    query = re.sub(r'[^\w\s\.\,\$\(\)\-]', '', query)
    query = re.sub(r'(\d+),(\d{2})\.(\d{2})', r'\1\2.\3', query)
    query = re.sub(r'(\d+\.\d{2})Net(\d+\.\d{2})', r'Gross $\1, Net $\2', query)
    query = re.sub(r'(\d+\.\d{2})\s*perhour', r'$\1 per hour', query, flags=re.IGNORECASE)
    query = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4\5', query)
    query = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4', query)
    query = re.sub(r'CPRPreviewSCA\s*\((\d+)\)\.pdf', r'CPRPreviewSCA_\1.pdf', query)
    logger.debug(f"Cleaned query: {query}")
    return query

async def build_chat_context(results: List[SearchResult]) -> str:
    """Build clean, pre-processed context with source tracking"""
    context = []
    file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}
    
    for result in results:
        # First pass: Clean basic formatting
        content = re.sub(r'[\n\r\t]', ' ', result.content).strip()
        content = re.sub(r'\s+', ' ', content)
        
        # Second pass: Fix number ranges and currency
        content = re.sub(r'(\d+)\s*([,.])\s*(\d+)', r'\1\2\3', content)
        content = re.sub(r'(\d+)\s*([a-zA-Z]+)\s*(\d+)', r'\1 \2 \3', content)
        content = re.sub(r'\$\s*(\d+)', r'$\1', content)
        
        # Format with source metadata
        context.append(
            f"DOCUMENT: {file_map.get(result.document_id, 'Unknown')}\n"
            f"SECTION: {result.chunk_index}\n"
            f"CONTENT: {content}\n"
            f"KEY_ENTITIES: {', '.join(result.entities) if result.entities else 'None'}"
        )
    
    # Ensure we don't exceed token limits
    return "\n\n---\n\n".join(context)[:7500]  # Conservative limit

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_coherent_response(query: str, context: str) -> str:
    """Generate properly formatted response with strict content separation"""
    validate_provider_settings()
    
    # Initialize client based on provider
    try:
        if settings.openai_enabled:
            client = OpenAI(api_key=settings.openai_api_key)
        else:
            client = None  # For Ollama, we'll use requests directly
    except Exception as e:
        logger.error(f"Failed to initialize client: {str(e)}")
        raise HTTPException(status_code=500, detail="Service configuration error")

    system_prompt = """You are a professional document analyst. STRICTLY follow these rules:

        1. CONTENT RULES:
        - Never copy text verbatim from documents
        - Always reformat numbers as continuous strings (e.g., $1,000,000-$3,000,000)
        - Never put numbers/currency on separate lines

        2. FORMATTING RULES:
        - Replace all newlines between numbers with hyphens
        - Remove spaces in currency values ($1M not $ 1 M)
        - Ensure ranges use proper format (X-Y not X newline Y)

        3. OUTPUT STRUCTURE:
        1. Overview paragraph
        2. Bullet points (max 5)
        3. Source reference"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

RESPONSE REQUIREMENTS:
1. Start with "This document discusses..." 
2. Include 3-5 key points as bullet points
3. Format all numbers/currency properly
4. End with "Source: [filename], section [X]" if available"""

    try:
        if settings.openai_enabled:
            if not client:
                raise ValueError("OpenAI client not initialized")
            response = client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=settings.max_completion_tokens,
                temperature=0.3
            )
            response_text = response.choices[0].message.content.strip()
        else:
            response = requests.post(
                f"http://{settings.ollama_host}:{settings.ollama_port}/v1/chat/completions",
                json={
                    "model": settings.ollama_chat_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": settings.max_completion_tokens,
                    "temperature": 0.3
                },
                timeout=30  # Added timeout
            )
            response.raise_for_status()
            response_text = response.json()["choices"][0]["message"]["content"].strip()
        
        return enforce_response_quality(response_text)
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=502, detail="Service temporarily unavailable")
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Service configuration error")
    except Exception as e:
        logger.error(f"Unexpected error in response generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response")
    
# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# API Endpoints
@app.post("/process_file", response_model=Dict[str, str])
async def process_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    """Process uploaded file and extract knowledge"""
    validate_provider_settings()
    if file.size > settings.max_document_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.max_document_size//(1024*1024)}MB limit"
        )

    existing_file = next(
        (f for f in state_manager.file_metadata
         if f['filename'] == file.filename and f['user_id'] == user_id),
        None
    )
    if existing_file:
        logger.info(f"File {file.filename} already exists for user {user_id}")
        return {
            "status": "success",
            "file_id": existing_file['file_id'],
            "filename": file.filename
        }

    file_ext = os.path.splitext(file.filename)[1].lower()
    file_id = str(uuid.uuid4())
    temp_path = Path(settings.temp_upload_dir) / f"{file_id}{file_ext}"

    try:
        file_content = await file.read()
        temp_path.write_bytes(file_content)

        if file_ext in settings.supported_extensions['pdfs']:
            logger.info(f"Processing PDF {file.filename} with OCR")
            markdown_content = ocr_processor.process_pdf(str(temp_path))
        else:
            converter = get_file_converter(file_ext)
            if not converter:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}"
                )
            markdown_content = converter(str(temp_path))

        is_ocr_likely = file_ext in settings.supported_extensions['images'] or file_ext in settings.supported_extensions['pdfs']
        if is_ocr_likely:
            cleaned_markdown = await preprocess_ocr_text(markdown_content)
        else:
            cleaned_markdown = text_processor.clean_markdown(markdown_content)

        chunk_embeddings = await text_processor.generate_embeddings(cleaned_markdown)
        if not chunk_embeddings:
            raise ValueError("No embeddings generated for the document")

        for i, (chunk_text, embedding) in enumerate(chunk_embeddings):
            chunk_id = str(uuid.uuid5(uuid.UUID(file_id), f"chunk_{i}"))
            entities = await text_processor.extract_entities(chunk_text)
            relationships = await text_processor.extract_relationships({"content": chunk_text, "chunk_index": i})

            logger.debug(f"Chunk {i} entities: {entities}")
            logger.debug(f"Chunk {i} relationships: {relationships}")

            qdrant_handler.store_chunk(
                document_id=file_id,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                embedding=embedding,
                metadata={
                    "filename": file.filename,
                    "user_id": user_id,
                    "chunk_index": i,
                    "parent_section": text_processor._extract_section(chunk_text),
                    "entities": entities,
                    "relationships": relationships
                }
            )

            for entity in entities:
                knowledge_graph.add_relationship(
                    source=entity,
                    target=file_id,
                    relationship="appears_in"
                )

            for rel in relationships:
                knowledge_graph.add_relationship(
                    source=rel['subject'],
                    target=rel['object'],
                    relationship=rel['predicate']
                )

        state_manager.file_metadata.append({
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_ext,
            "upload_date": datetime.datetime.now().isoformat(),
            "content": base64.b64encode(file_content).decode(),
            "markdown_content": cleaned_markdown,
            "user_id": user_id,
            "size": file.size,
            "checksum": hashlib.md5(file_content).hexdigest(),
            "chunks": len(chunk_embeddings)
        })

        state_manager.save()
        logger.info(f"Processed file: {file.filename} (ID: {file_id}) with {len(chunk_embeddings)} chunks")
        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"File processing failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.error(f"Temp file cleanup failed for {temp_path}: {str(e)}")

@app.post("/search", response_model=Dict[str, Any])
async def search_documents(
    query: str = Form(...),
    user_id: str = Form(...),
    file_ids: Optional[List[str]] = Form(None),
    limit: int = Form(5),
    use_graph: bool = Form(True),
    api_key: str = Depends(validate_api_key)
):
    """Search documents with dual-level retrieval"""
    validate_provider_settings()
    try:
        cleaned_query = clean_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Cleaned query for search: {cleaned_query}")

        entity_results = []
        if use_graph:
            entities = await text_processor.extract_entities(cleaned_query)
            for entity in entities:
                related_entities = knowledge_graph.find_related_entities(entity)
                if related_entities:
                    entity_results.extend(
                        await qdrant_handler.search_entities(
                            entities=related_entities,
                            user_id=user_id,
                            file_ids=file_ids,
                            limit=limit
                        )
                    )

        query_chunks = split_large_text(cleaned_query)
        vector_results = []

        for chunk in query_chunks:
            embedding = generate_embeddings_batch([chunk])[0]
            query_filter = {
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                ]
            }
            if file_ids:
                query_filter["must"].append(
                    {"key": "document_id", "match": {"any": file_ids}}
                )

            results = qdrant_handler.client.search(
                collection_name=settings.qdrant_collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=limit
            )
            vector_results.extend(results)

        combined_results = rank_results(entity_results, vector_results, limit)

        return {
            "status": "success",
            "results": format_search_results(combined_results)
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Dict[str, Any])
async def chat_with_documents(
    query: str = Form(...),
    user_id: str = Form(...),
    file_ids: Optional[List[str]] = Form(None),
    chat_id: Optional[str] = Form(None),
    api_key: str = Depends(validate_api_key)
):
    """Chat endpoint with enforced response quality"""
    try:
        cleaned_query = clean_query(query)
        search_results = await search_documents(
            query=cleaned_query,
            user_id=user_id,
            file_ids=file_ids,
            limit=5,  # Fewer but more relevant results
            use_graph=True
        )

        if not search_results["results"]:
            return {
                "response": "No relevant information found in the documents.",
                "chat_id": chat_id or str(uuid.uuid4()),
                "sources": []
            }

        context = await build_chat_context(search_results["results"])
        response = generate_coherent_response(cleaned_query, context)
        response = enforce_response_quality(response)

        chat_id = chat_id or str(uuid.uuid4())
        if chat_id not in state_manager.chat_sessions:
            state_manager.chat_sessions[chat_id] = {
                "chat_id": chat_id,
                "user_id": user_id,
                "messages": [],
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "document_ids": file_ids or []
            }

        state_manager.chat_sessions[chat_id]["messages"].append({
            "role": "user",
            "content": query,
            "timestamp": datetime.datetime.now().isoformat()
        })
        state_manager.chat_sessions[chat_id]["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        state_manager.chat_sessions[chat_id]["updated_at"] = datetime.datetime.now().isoformat()

        state_manager.save()
        return {
            "response": response,
            "chat_id": chat_id,
            "sources": search_results["results"]
        }

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    """List uploaded documents for a user"""
    try:
        user_docs = [
            f for f in state_manager.file_metadata
            if f.get("user_id") == user_id
        ]
        return {
            "status": "success",
            "documents": [
                {
                    "file_id": f["file_id"],
                    "filename": f["filename"],
                    "file_type": f["file_type"],
                    "upload_date": f["upload_date"],
                    "size": f["size"]
                }
                for f in user_docs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@app.delete("/documents/{file_id}", response_model=Dict[str, str])
async def delete_document(
    file_id: str,
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    """Delete a document and its chunks"""
    try:
        await qdrant_handler.delete_by_document_id(file_id)
        state_manager.file_metadata = [
            f for f in state_manager.file_metadata
            if f["file_id"] != file_id
        ]
        state_manager.save()
        logger.info(f"Deleted document: {file_id}")
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        logger.error(f"Failed to delete document {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@app.get("/preview/{file_id}")
async def preview_file(file_id: str, user_id: str, api_key: str = Depends(validate_api_key)):
    """Stream file content for preview"""
    file_meta = next((f for f in state_manager.file_metadata if f['file_id'] == file_id and f['user_id'] == user_id), None)
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    mime_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".heic": "image/heic",
        ".webp": "image/webp",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".odt": "application/vnd.oasis.opendocument.text",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".ods": "application/vnd.oasis.opendocument.spreadsheet",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".rtf": "application/rtf"
    }
    return StreamingResponse(
        BytesIO(base64.b64decode(file_meta['content'])),
        media_type=mime_map.get(file_meta['file_type'], "application/octet-stream")
    )

@app.get("/knowledge_graph", response_model=Dict[str, Any])
async def get_knowledge_graph(
    user_id: str,
    file_id: Optional[str] = None,
    api_key: str = Depends(validate_api_key)
):
    """Retrieve knowledge graph data for visualization"""
    try:
        nodes = []
        edges = []
        file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}

        txn = knowledge_graph.client.txn(read_only=True)
        try:
            query = """
                query {
                    entities(func: eq(type, "entity")) {
                        uid
                        name
                        type
                        appears_in {
                            uid
                            document_id
                        }
                        relationship {
                            uid
                            name
                            type
                            predicate
                            weight
                        }
                    }
                    documents(func: has(document_id)) {
                        uid
                        document_id
                    }
                }
            """
            if file_id:
                query = f"""
                    query {{
                        entities(func: eq(type, "entity")) @filter(has(appears_in)) {{
                            uid
                            name
                            type
                            appears_in @filter(eq(document_id, "{file_id}")) {{
                                uid
                                document_id
                            }}
                            relationship {{
                                uid
                                name
                                type
                                predicate
                                weight
                            }}
                        }}
                        documents(func: eq(document_id, "{file_id}")) {{
                            uid
                            document_id
                        }}
                    }}
                """
            res = txn.query(query)
            data = json.loads(res.json)

            for entity in data.get("entities", []):
                nodes.append({
                    "id": entity["uid"],
                    "label": entity["name"],
                    "type": "entity"
                })
                for rel in entity.get("relationship", []):
                    if rel.get("name"):
                        nodes.append({
                            "id": rel["uid"],
                            "label": rel["name"],
                            "type": rel.get("type", "entity")
                        })
                        edges.append({
                            "from": entity["uid"],
                            "to": rel["uid"],
                            "label": rel["predicate"],
                            "weight": rel.get("weight", 1.0)
                        })
                for doc in entity.get("appears_in", []):
                    if doc.get("document_id"):
                        edges.append({
                            "from": entity["uid"],
                            "to": doc["uid"],
                            "label": "appears_in",
                            "weight": 1.0
                        })

            for doc in data.get("documents", []):
                nodes.append({
                    "id": doc["uid"],
                    "label": file_map.get(doc["document_id"], doc["document_id"]),
                    "type": "document"
                })

            return {
                "status": "success",
                "nodes": nodes,
                "edges": edges
            }
        finally:
            txn.discard()
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))