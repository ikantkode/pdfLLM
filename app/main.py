import os
import uuid
import json
import logging
import base64
import datetime
import hashlib
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
            combined.append(result)
    
    return sorted(
        combined,
        key=lambda x: x.score if hasattr(x, 'score') else x.get('score', 0.0),
        reverse=True
    )[:limit]

async def build_chat_context(results: List[SearchResult]) -> str:
    """Build context from search results with token limits"""
    context = ""
    token_count = 0
    file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}
    
    for result in results:
        content = result.content
        tokens = len(tokenizer.encode(content))
        
        if token_count + tokens > settings.max_completion_tokens * 0.6:
            break
            
        context += f"Document: {file_map.get(result.document_id, 'Unknown')}\n"
        context += f"Section: {result.chunk_index}\n"
        context += f"Entities: {', '.join(result.entities)}\n"
        context += f"Relationships: {', '.join(['{} {} {}'.format(rel['subject'], rel['predicate'], rel['object']) for rel in result.relationships])}\n"
        context += f"Content: {content}\n\n"
        token_count += tokens
        
    return context

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_coherent_response(query: str, context: str) -> str:
    """Generate response with proper token management"""
    prompt = (
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        "Answer the query concisely based on the context. Cite the document name and section number. "
        "Include-relevant entities or relationships only if they directly relate to the answer. "
        "Return a plain-text response with no special characters, newlines between characters, or repeated text. "
        f"Keep response under {settings.max_completion_tokens} tokens. "
        "Example: The requisition amount for PS 54X is $29,825.80, noted in Section 1 of AMB PS 54X AIA3.pdf under 'CURRENT PAYMENT DUE'."
    )
    
    try:
        response = openai.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that answers queries based on document context. Provide concise, plain-text responses, citing sources clearly. Avoid special characters, excessive newlines, or repeating context."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=settings.max_completion_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
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
    file_id: Optional[str] = Form(None),
    limit: int = Form(5),
    use_graph: bool = Form(True),
    api_key: str = Depends(validate_api_key)
):
    """Search documents with dual-level retrieval"""
    try:
        # Entity-based retrieval from knowledge graph
        entity_results = []
        if use_graph:
            entities = await text_processor.extract_entities(query)
            for entity in entities:
                related_entities = knowledge_graph.find_related_entities(entity)
                if related_entities:
                    entity_results.extend(
                        await qdrant_handler.search_entities(
                            entities=related_entities,
                            user_id=user_id,
                            file_id=file_id,
                            limit=limit
                        )
                    )

        # Vector-based retrieval
        query_chunks = split_large_text(query)
        vector_results = []
        
        for chunk in query_chunks:
            embedding = generate_embeddings_batch([chunk])[0]
            results = qdrant_handler.client.search(
                collection_name=settings.qdrant_collection,
                query_vector=embedding,
                query_filter={
                    "must": [
                        {"key": "user_id", "match": {"value": user_id}},
                        *([{"key": "document_id", "match": {"value": file_id}}] if file_id else [])
                    ]
                },
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
        # Perform dual-level retrieval
        search_results = await search_documents(
            query=query,
            user_id=user_id,
            file_id=file_ids[0] if file_ids else None,
            limit=5,
            use_graph=True
        )
        
        if not search_results["results"]:
            return {
                "response": "No relevant information found in documents.",
                "chat_id": chat_id or str(uuid.uuid4())
            }

        # Build context with chunked approach
        context = await build_chat_context(search_results["results"])
        
        # Generate response
        response = generate_coherent_response(query, context)
        
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
    return StreamingResponse(BytesIO(base64.b64decode(file_meta['content'])), media_type=mime_map.get(file_meta['file_type'], "application/octet-stream"))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)