import os
import uuid
import json
import logging
import base64
import datetime
import hashlib
import re
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
import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from pydantic import BaseModel
from config import settings
from utils.qdrant_handler import QdrantHandler
from utils.text_processor import TextProcessor
from converters import (
    image_converter,
    doc_converter,
    excel_converter,
    pdf_converter,
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
    if api_key != settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Initialize components
try:
    qdrant_handler = QdrantHandler(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection
    )
    logger.info("Qdrant connection established")
except Exception as e:
    logger.error(f"Qdrant connection failed: {str(e)}")
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
        self.graph = nx.DiGraph()
        self.entity_index = {}

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship: str,
        weight: float = 1.0
    ) -> None:
        """Add a relationship between entities"""
        logger.debug(f"Adding relationship: {source} -> {relationship} -> {target}")
        if source not in self.graph:
            self.graph.add_node(source, type='entity')
        if target not in self.graph:
            self.graph.add_node(target, type='entity')
        self.graph.add_edge(
            source,
            target,
            relationship=relationship,
            weight=weight
        )

    def find_related_entities(
        self,
        entity: str,
        depth: int = settings.max_graph_depth
    ) -> List[str]:
        """Find related entities within specified depth"""
        if entity not in self.graph:
            return []
        return list(nx.single_source_shortest_path_length(
            self.graph,
            entity,
            cutoff=depth
        ).keys())

    def save(self) -> None:
        """Save graph to file"""
        data = nx.node_link_data(self.graph)
        try:
            with open(settings.graph_file, 'w') as f:
                json.dump(data, f)
            logger.info("Knowledge graph saved successfully")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {str(e)}")

    def load(self) -> None:
        """Load graph from file"""
        if os.path.exists(settings.graph_file):
            try:
                with open(settings.graph_file, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info("Knowledge graph loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {str(e)}")

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
            knowledge_graph.save()
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
                knowledge_graph.load()
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
    elif file_ext in settings.supported_extensions['pdfs']:
        return pdf_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['text']:
        return txt_converter.convert_to_markdown
    return None

def split_large_text(text: str, max_tokens: int = settings.max_embedding_tokens) -> List[str]:
    """Split text into chunks that fit within token limits"""
    tokens = tokenizer.encode(text)
    return [
        tokenizer.decode(tokens[i:i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings with batch processing and retry logic"""
    try:
        response = openai.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

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
            score += 0.1 * len(result.payload.get('entities', []))  # Boost for entities
            score += 0.2 * len(result.payload.get('relationships', []))  # Boost for relationships
            result.score = score
            combined.append(result)

    return sorted(
        combined,
        key=lambda x: x.score if hasattr(x, 'score') else x.get('score', 0.0),
        reverse=True
    )[:limit]

def clean_text_for_context(text: str) -> str:
    """Clean text to handle OCR issues and improve coherence"""
    # Normalize whitespace but preserve line structure
    text = re.sub(r'[ \t]+', ' ', text)  # Only normalize spaces/tabs
    text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading spaces after newlines
    text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing spaces before newlines
    text = text.strip()
    
    # Remove excessive special characters but be conservative
    text = re.sub(r'([^\w\s])\1{2,}', r'\1', text)  # Remove 3+ repeated special chars
    
    # Remove repeated words/phrases but be more conservative
    text = re.sub(r'\b(\w+)\s+\1\s+\1\b', r'\1', text, flags=re.IGNORECASE)  # Only remove 3+ repeats
    
    # Remove clearly problematic characters, but preserve more formatting
    text = re.sub(r'[^\w\s\.\,\$\(\)\-\n\r\t:;]', '', text)
    
    # Fix malformed numbers (e.g., "1,98.16" -> "198.16")
    text = re.sub(r'(\d+),(\d{2})\.(\d{2})', r'\1\2.\3', text)
    
    # Filter out gibberish (require at least 3 meaningful words)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        words = line.split()
        if len([w for w in words if len(w) > 2]) >= 3:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def clean_query(query: str) -> str:
    """Clean user query to handle OCR-like noise and payroll-specific formatting"""
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove excessive special characters
    query = re.sub(r'([^\w\s])\1+', r'\1', query)
    
    # Remove repeated words/phrases
    query = re.sub(r'\b(\w+)\s+\1\b', r'\1', query, flags=re.IGNORECASE)
    
    # Remove invalid characters, preserving common punctuation and currency
    query = re.sub(r'[^\w\s\.\,\$\(\)\-]', '', query)
    
    # Fix malformed numbers (e.g., "1,98.16" -> "1988.16")
    query = re.sub(r'(\d+),(\d{2})\.(\d{2})', r'\1\2.\3', query)
    
    # Correct payroll-specific formats (e.g., "7,068.92Net5,079.50" -> "Gross $7068.92, Net $5079.50")
    query = re.sub(r'(\d+\.\d{2})Net(\d+\.\d{2})', r'Gross $\1, Net $\2', query)
    
    # Fix per-hour rates (e.g., "5.05perhour" -> "$5.05 per hour")
    query = re.sub(r'(\d+\.\d{2})\s*perhour', r'$\1 per hour', query, flags=re.IGNORECASE)
    
    # Reconstruct fragmented names (e.g., "D z i u b a" -> "Dziuba")
    query = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4\5', query)
    query = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4', query)
    
    # Fix document names (e.g., "CPRPreviewSCA (1).pdf" -> "CPRPreviewSCA_1.pdf")
    query = re.sub(r'CPRPreviewSCA\s*\((\d+)\)\.pdf', r'CPRPreviewSCA_\1.pdf', query)
    
    logger.debug(f"Cleaned query: {query}")
    return query

async def build_chat_context(results: List[SearchResult]) -> str:
    """Build context from search results with token limits and OCR cleanup"""
    context = ""
    token_count = 0
    file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}
    seen_content = set()

    for result in results:
        content = result.content
        # Clean content to handle OCR issues
        cleaned_content = clean_text_for_context(content)
        
        # Skip empty or low-quality content
        if not cleaned_content:
            logger.debug(f"Skipped low-quality chunk: {content[:100]}...")
            continue
            
        # Skip duplicate content
        if cleaned_content in seen_content:
            logger.debug(f"Skipped duplicate chunk: {cleaned_content[:100]}...")
            continue
        seen_content.add(cleaned_content)

        tokens = len(tokenizer.encode(cleaned_content))
        if token_count + tokens > settings.max_completion_tokens * 0.8:
            logger.debug(f"Context truncated at {token_count} tokens to stay within limit")
            break

        context += f"Document: {file_map.get(result.document_id, 'Unknown')}\n"
        context += f"Section: {result.chunk_index}\n"
        context += f"Entities: {', '.join(result.entities)}\n"
        context += f"Relationships: {', '.join(['{} {} {}'.format(rel['subject'], rel['predicate'], rel['object']) for rel in result.relationships])}\n"
        context += f"Content: {cleaned_content}\n\n"
        token_count += tokens

    logger.debug(f"Built context with {token_count} tokens: {context[:500]}...")
    return context

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_coherent_response(query: str, context: str) -> str:
    """Generate response with proper token management and assistant-like behavior"""
    prompt = (
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        "You are a professional personal assistant tasked with providing a clear, concise, and accurate response based solely on the provided context and query. "
        "The context and query may contain noisy or poorly formatted text due to OCR errors (e.g., jumbled characters, repeated fragments, misaligned formatting, or malformed numbers like '1,98.16' or '7,068.92Net5,079.50'). "
        "Follow these instructions:\n"
        "1. Structure the response in a well-organized manner with complete sentences and proper grammar.\n"
        "2. Avoid repetition of words, phrases, or numbers unless necessary for clarity.\n"
        "3. Correct and format numerical values (e.g., '1,98.16' → '$1,988.16', '7,068.92Net5,079.50' → 'Gross $7,068.92, Net $5,079.50', '5.05perhour' → '$5.05 per hour').\n"
        "4. Cite the document name and section number for each piece of information used (e.g., 'CPRPreviewSCA_1.pdf (Section 0)').\n"
        "5. Include relevant entities or relationships only if they directly contribute to answering the query.\n"
        "6. Do not speculate or include information not present in the context or query.\n"
        "7. For queries asking to 'explain' payroll documents, provide a narrative summary of their content, purpose, and key details, followed by a structured list or table of payroll details (e.g., Employee, Role, Hours, Rate, Gross Pay, Net Pay).\n"
        "8. For queries asking for total expenditure, calculate the sum of gross and net pay across all documents, validating numbers against the context and query, and present the totals clearly.\n"
        "9. Keep the response under {settings.max_completion_tokens} tokens.\n"
        "10. Handle noisy input by prioritizing meaningful information, ignoring jumbled characters, repeated fragments, or formatting errors.\n"
        "11. Reconstruct malformed text or numbers (e.g., 'D z i u b a' → 'Dziuba', '1,98.16' → '$1,988.16') and validate against context where possible.\n"
        "12. If hours or rates are missing, estimate them using gross pay and typical rates from the context, noting assumptions.\n"
        "13. If the context or query is unclear, reconstruct the most likely intended meaning and note limitations (e.g., 'Some details may be incomplete due to OCR errors').\n\n"
        "Example response for 'Explain the document and tell me how much money was spent in total':\n"
        "The documents 'CPRPreviewSCA_6.pdf' and 'CPRPreviewSCA_5.pdf' are certified payroll reports from AMB Contractors Inc. for the P.S.54 - BRONX project, detailing employee wages. Below is a summary of the payroll details:\n\n"
        "| Employee | Role | Hours | Rate | Gross Pay | Net Pay | Source |\n"
        "|----------|------|-------|------|-----------|---------|--------|\n"
        "| Zenovii Dziuba | Laborer | 16 | $94.43 | $1,510.88 | $1,143.20 | CPRPreviewSCA_6.pdf (Section 0) |\n"
        "| Ermilo Lopez | Laborer | 8 | $94.43 | $755.44 | $608.65 | CPRPreviewSCA_6.pdf (Section 0) |\n"
        "| Zenovii Dziuba | Laborer | 30 | $94.43 | $2,832.90 | $1,988.16 | CPRPreviewSCA_5.pdf (Section 0) |\n"
        "| Ermilo Lopez | Laborer | 30 | $94.43 | $2,832.90 | $1,988.16 | CPRPreviewSCA_5.pdf (Section 0) |\n\n"
        "Total expenditure: Gross Pay $7,932.12, Net Pay $5,728.17. Some details may be incomplete due to OCR errors."
    )

    try:
        response = openai.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional personal assistant. Provide clear, concise, and accurate responses based strictly on the provided context and query. "
                        "Handle noisy text from OCR errors (e.g., jumbled characters, repeated fragments, malformed numbers like '1,98.16' or '7,068.92Net5,079.50') by prioritizing coherent information and reconstructing meaning. "
                        "Structure responses with complete sentences, proper grammar, and correct spelling. "
                        "Format numbers correctly (e.g., '$1,988.16', '$5.05 per hour') and cite sources (document name and section). "
                        "For payroll-related queries, present details in a clear list or table format, including Employee, Role, Hours, Rate, Gross Pay, and Net Pay. "
                        "Calculate total expenditure (gross and net) when requested, validating against context and query. "
                        "Estimate missing hours or rates using context data, noting assumptions. "
                        "Do not speculate or add information beyond the context or query. "
                        "Note limitations due to OCR errors if applicable."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=settings.max_completion_tokens,
            temperature=0.3
        )
        response_text = response.choices[0].message.content.strip()
        
        # MUCH more gentle post-processing - only fix obvious OCR issues
        # Remove excessive whitespace but preserve line breaks and formatting
        response_text = re.sub(r'[ \t]+', ' ', response_text)  # Only normalize spaces/tabs, keep newlines
        response_text = re.sub(r'\n[ \t]+', '\n', response_text)  # Remove leading spaces after newlines
        response_text = re.sub(r'[ \t]+\n', '\n', response_text)  # Remove trailing spaces before newlines
        
        # Fix obvious OCR number errors but be very conservative
        response_text = re.sub(r'(\d+),(\d{2})\.(\d{2})', r'\1\2.\3', response_text)  # Fix "1,98.16" -> "198.16"
        response_text = re.sub(r'(\d+\.\d{2})Net(\d+\.\d{2})', r'Gross $\1, Net $\2', response_text)  # Fix joined numbers
        
        # Remove only clearly problematic characters, preserve formatting
        response_text = re.sub(r'[^\w\s\.\,\$\(\)\-\|\n\r\t:;]', '', response_text)
        
        # DON'T remove "repeated" words - this breaks legitimate formatting!
        # DON'T collapse all whitespace - this destroys table formatting!
        
        logger.debug(f"Generated response: {response_text[:500]}...")
        return response_text
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        raise

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
    if file.size > settings.max_document_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.max_document_size//(1024*1024)}MB limit"
        )

    # Check for existing file
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
        # Save and convert file
        file_content = await file.read()
        temp_path.write_bytes(file_content)

        converter = get_file_converter(file_ext)
        if not converter:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}"
            )

        markdown_content = converter(str(temp_path))

        # Process content with enhanced chunking
        cleaned_markdown = text_processor.clean_markdown(markdown_content)
        chunks = text_processor.chunk_text(cleaned_markdown)

        # Process in batches for large documents
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Generate embeddings and extract knowledge
            batch = await text_processor.generate_embeddings(batch)

            for chunk in batch:
                chunk['document_id'] = file_id
                chunk['user_id'] = user_id

                # Add to vector store
                await qdrant_handler.save_chunk(chunk, user_id)

                # Index entities and relationships in knowledge graph
                for entity in chunk.get('entities', []):
                    knowledge_graph.add_relationship(
                        source=entity,
                        target=file_id,
                        relationship="appears_in"
                    )

                for rel in chunk.get('relationships', []):
                    knowledge_graph.add_relationship(
                        source=rel['subject'],
                        target=rel['object'],
                        relationship=rel['predicate']
                    )

        # Store metadata
        state_manager.file_metadata.append({
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_ext,
            "upload_date": datetime.datetime.now().isoformat(),
            "content": base64.b64encode(file_content).decode(),
            "markdown_content": markdown_content,
            "user_id": user_id,
            "size": file.size,
            "checksum": hashlib.md5(file_content).hexdigest()
        })

        state_manager.save()
        logger.info(f"Processed file: {file.filename} (ID: {file_id})")
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
    try:
        # Clean query before processing
        cleaned_query = clean_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Cleaned query for search: {cleaned_query}")
        
        # Entity-based retrieval from knowledge graph
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

        # Vector-based retrieval
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

        # Combine and rank results
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
    """Chat with document context"""
    try:
        # Clean query before processing
        cleaned_query = clean_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Cleaned query for chat: {cleaned_query}")

        # Perform dual-level retrieval
        search_results = await search_documents(
            query=cleaned_query,
            user_id=user_id,
            file_ids=file_ids,
            limit=10,
            use_graph=True
        )

        if not search_results["results"]:
            return {
                "response": "No relevant information found in documents.",
                "chat_id": chat_id or str(uuid.uuid4()),
                "sources": []
            }

        # Build context with chunked approach
        context = await build_chat_context(search_results["results"])

        # Generate response
        response = generate_coherent_response(cleaned_query, context)

        # Clean response to avoid formatting issues
        response = ' '.join(response.split()).strip()

        # Update chat history
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
            "content": query,  # Store original query for history
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
    return StreamingResponse(BytesIO(base64.b64decode(file_meta['content'])), media_type=mime_map.get(file_meta['file_type'], "application/octet-stream"))

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

        # Filter nodes and edges by user_id and optionally file_id
        for node, data in knowledge_graph.graph.nodes(data=True):
            node_type = data.get('type', 'entity')
            if node_type == 'entity' and (not file_id or any(
                edge[1] == file_id for edge in knowledge_graph.graph.edges(node)
            )):
                nodes.append({
                    "id": node,
                    "label": node,
                    "type": node_type
                })

        # Include document nodes if they match file_id or user_id
        if file_id:
            if file_id in file_map:
                nodes.append({
                    "id": file_id,
                    "label": file_map[file_id],
                    "type": "entity"
                })
        else:
            for file in state_manager.file_metadata:
                if file['user_id'] == user_id:
                    nodes.append({
                        "id": file['file_id'],
                        "label": file['filename'],
                        "type": "entity"
                    })

        # Collect edges
        for source, target, data in knowledge_graph.graph.edges(data=True):
            if file_id and target != file_id and source != file_id:
                continue
            if any(f['file_id'] == target and f['user_id'] == user_id for f in state_manager.file_metadata) or \
               any(f['file_id'] == source and f['user_id'] == user_id for f in state_manager.file_metadata):
                edges.append({
                    "from": source,
                    "to": target,
                    "label": data.get("relationship", "related_to"),
                    "weight": data.get("weight", 1.0)
                })

        logger.info(f"Retrieved knowledge graph for user_id={user_id}, file_id={file_id}")
        return {
            "status": "success",
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)