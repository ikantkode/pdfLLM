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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.encoders import jsonable_encoder
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel
from app.config import settings
from app.utils.qdrant_handler import QdrantHandler
from app.utils.text_processor import TextProcessor
from app.utils.ocr_processor import OCRProcessor
from app.converters import image_converter, doc_converter, excel_converter, txt_converter
from app.celery_app import celery_app
from app.utils.helpers import preprocess_ocr_text, classify_document, get_db_connection
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny
from psycopg2.extras import Json  # Added to fix NameError

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
        logger.error(f"Invalid API key provided: {api_key[:4]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Check provider configuration
def validate_provider_settings():
    if not settings.openai_enabled:
        logger.error("OpenAI is not enabled. Please set OPENAI_ENABLED to true.")
        raise HTTPException(
            status_code=400,
            detail="Invalid configuration: OPENAI_ENABLED must be set to true."
        )

# Initialize OCR processor
try:
    ocr_processor = OCRProcessor()
    logger.info("OCRProcessor initialized successfully")
except Exception as e:
    logger.error(f"OCRProcessor initialization failed: {str(e)}")
    raise

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
    content: Optional[str] = None
    markdown_content: Optional[str] = None
    user_id: str
    size: int
    checksum: str
    category: Optional[str] = None
    status: str

class ChatMessage(BaseModel):
    message_id: str
    chat_id: str
    role: str
    content: str
    timestamp: str

class ChatSession(BaseModel):
    chat_id: str
    user_id: str
    created_at: str
    updated_at: str
    document_ids: List[str]
    module: Optional[str] = None

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
    category: Optional[str] = None

class Prompt(BaseModel):
    id: int
    category: str
    prompt: str
    created_at: str
    updated_at: str
    user_id: str

# Helper functions
def get_file_converter(file_ext: str):
    if file_ext in settings.supported_extensions['images']:
        return image_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['documents']:
        return doc_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['spreadsheets']:
        return excel_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['text']:
        return txt_converter.convert_to_markdown
    return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    validate_provider_settings()
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            input=texts,
            model=settings.openai_embedding_model,
            dimensions=1024
        )
        embeddings = [item.embedding for item in response.data]
        logger.info(f"OpenAI embeddings generated with {len(embeddings[0])} dimensions")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

def clean_response(text: str) -> str:
    text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)
    text = re.sub(r'(?<=\d)\n(?=\d)', '', text)
    text = re.sub(r'(?<=\$)\n(?=\d)', '', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3\4', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\b', r'\1\2\3', text)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', r'\1\2', text)
    text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)
    text = re.sub(r'\$\s+(\d)', r'$\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    text = '\n\n'.join(
        f"{p[0].upper()}{p[1:]}" if p and p[0].islower() else p
        for p in paragraphs
    )
    return text.strip()

def format_search_results(results: List, file_map: Dict[str, Dict]) -> List[SearchResult]:
    return [
        SearchResult(
            chunk_id=str(r.id),
            document_id=r.payload.get("document_id", "N/A"),
            filename=file_map.get(r.payload.get("document_id", "N/A"), {}).get("filename", "Unknown"),
            parent_section=r.payload.get("parent_section", "N/A"),
            chunk_index=r.payload.get("chunk_index", 0),
            content=r.payload.get("content", "N/A"),
            entities=r.payload.get("entities", []),
            relationships=r.payload.get("relationships", []),
            score=r.score,
            category=r.payload.get("category")
        )
        for r in results
    ]

def rank_results(vector_results: List, limit: int, file_ids: Optional[List[str]] = None) -> List:
    seen = set()
    combined = []
    doc_counts = {}

    for result in vector_results:
        result_id = result.id
        if result_id not in seen:
            seen.add(result_id)
            score = result.score
            score += 0.1 * len(result.payload.get('entities', []))
            score += 0.2 * len(result.payload.get('relationships', []))
            document_id = result.payload.get('document_id', 'N/A')
            if file_ids and document_id in file_ids:
                doc_counts[document_id] = doc_counts.get(document_id, 0) + 1
                score += 1.0 / (doc_counts[document_id] + 1)
            result.score = score
            combined.append(result)

    sorted_results = sorted(combined, key=lambda x: x.score, reverse=True)
    if file_ids:
        final_results = []
        seen_docs = set()
        for result in sorted_results:
            doc_id = result.payload.get('document_id', 'N/A')
            if doc_id in file_ids and doc_id not in seen_docs:
                final_results.append(result)
                seen_docs.add(doc_id)
            remaining = [r for r in sorted_results if r not in final_results]
            final_results.extend(remaining[:limit - len(final_results)])
            return final_results[:limit]
    return sorted_results[:limit]

def clean_query(query: str) -> str:
    query = re.sub(r'[\n\r\t]', ' ', query).strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'([^\w\s])\1+', r'\1', query)
    query = re.sub(r'\b(\w+)\s+\1\b', r'\1', query, flags=re.IGNORECASE)
    query = re.sub(r'[^\w\s\.\,\$\(\)\-]', '', query)
    query = re.sub(r'(\d+),(\d{2})\.(\d{2})', r'\1\2.\3', query)
    query = re.sub(r'(\d+\.\d{2})\s*perhour', r'$\1 per hour', query, flags=re.IGNORECASE)
    return query

async def build_chat_context(results: List[SearchResult]) -> str:
    context = []
    for result in results:
        content = re.sub(r'[\n\r\t]', ' ', result.content).strip()
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'(\d+)\s*([,.])\s*(\d+)', r'\1\2\3', content)
        content = re.sub(r'(\d+)\s*([a-zA-Z]+)\s*(\d+)', r'\1 \2 \3', content)
        content = re.sub(r'\$\s*(\d+)', r'$\1', content)
        context.append(
            f"DOCUMENT: {result.filename}\n"
            f"SECTION: {result.chunk_index}\n"
            f"CONTENT: {content}\n"
            f"KEY_ENTITIES: {', '.join(result.entities) if result.entities else 'None'}\n"
            f"CATEGORY: {result.category or 'None'}"
        )
    return "\n\n---\n\n".join(context)[:10000]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_coherent_response(query: str, context: str, category: str, user_id: str) -> str:
    validate_provider_settings()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prompt FROM prompts WHERE category = %s AND user_id = %s",
                (category, user_id)
            )
            prompt_row = cur.fetchone()
            system_prompt = prompt_row[0] if prompt_row else "You are a document analyst."
    except Exception as e:
        logger.error(f"Failed to fetch prompt for category {category}, user {user_id}: {str(e)}")
        raise
    finally:
        conn.close()

    try:
        client = OpenAI(api_key=settings.openai_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail="Service configuration error")

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}"""

    try:
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
        return clean_response(response_text)
    except Exception as e:
        logger.error(f"Unexpected error in response generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

# API Endpoints
@app.post("/process_file", response_model=Dict[str, str])
async def process_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    category: Optional[str] = Form(None),
    api_key: str = Depends(validate_api_key)
):
    validate_provider_settings()
    if file.size > settings.max_document_size:
        logger.error(f"File {file.filename} size {file.size/(1024*1024):.2f}MB exceeds limit {settings.max_document_size/(1024*1024):.2f}MB")
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.max_document_size//(1024*1024)}MB limit"
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_id FROM file_metadata WHERE filename = %s AND user_id = %s",
                (file.filename, user_id)
            )
            existing_file = cur.fetchone()
            if existing_file:
                logger.info(f"File {file.filename} already exists for user {user_id}, file_id: {existing_file[0]}")
                return {
                    "status": "success",
                    "file_id": existing_file[0],
                    "filename": file.filename
                }

        file_id = str(uuid.uuid4())
        file_content = await file.read()
        file_ext = os.path.splitext(file.filename)[1].lower()
        checksum = hashlib.md5(file_content).hexdigest()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO file_metadata (file_id, filename, file_type, upload_date, content, user_id, size, checksum, category, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (file_id, file.filename, file_ext, datetime.datetime.now(), file_content, user_id, file.size, checksum, category, "pending")
            )
            conn.commit()

        celery_app.send_task(
            "app.celery_app.process_ocr",
            args=[file_id, user_id, file_ext, category],
            kwargs={"filename": file.filename}
        )
        logger.info(f"Queued OCR processing for file: {file.filename} (ID: {file_id})")
        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"File processing failed for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")
    finally:
        conn.close()

@app.post("/chat", response_model=Dict[str, Any])
async def chat_with_documents(
    query: str = Form(...),
    user_id: str = Form(...),
    file_ids: Optional[List[str]] = Form(None),
    chat_id: Optional[str] = Form(None),
    category: Optional[str] = Form("all"),
    api_key: str = Depends(validate_api_key)
):
    try:
        cleaned_query = clean_query(query)
        logger.debug(f"Processing chat query: {cleaned_query[:100]}... for user: {user_id}, category: {category}")
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                query_chunks = text_processor.chunk_text(cleaned_query)
                per_doc_limit = max(1, 5 // max(1, len(file_ids or [])))
                vector_results = []
                for chunk in query_chunks:
                    embedding = generate_embeddings_batch([chunk['content']])[0]
                    query_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
                    if file_ids:
                        query_filter.must.append(FieldCondition(key="document_id", match=MatchAny(any=file_ids)))
                    if category != "all":
                        query_filter.must.append(FieldCondition(key="category", match=MatchValue(value=category)))
                    results = qdrant_handler.client.search(
                        collection_name=settings.qdrant_collection,
                        query_vector=embedding,
                        query_filter=query_filter,
                        limit=per_doc_limit
                    )
                    vector_results.extend(results)

                cur.execute(
                    "SELECT file_id, filename, category FROM file_metadata WHERE file_id = ANY(%s) AND user_id = %s",
                    (file_ids or [], user_id)
                )
                file_map = {row[0]: {"filename": row[1], "category": row[2]} for row in cur.fetchall()}
                combined_results = rank_results(vector_results, 5, file_ids)
                search_results = format_search_results(combined_results, file_map)

                if not search_results:
                    logger.info(f"No relevant information found for query: {cleaned_query[:100]}...")
                    return {
                        "response": "No relevant information found in the documents.",
                        "chat_id": chat_id or str(uuid.uuid4()),
                        "sources": []
                    }

                context = await build_chat_context(search_results)
                response = generate_coherent_response(cleaned_query, context, category, user_id)

                chat_id = chat_id or str(uuid.uuid4())
                cur.execute(
                    "SELECT chat_id FROM chat_sessions WHERE chat_id = %s AND user_id = %s",
                    (chat_id, user_id)
                )
                if not cur.fetchone():
                    cur.execute(
                        """
                        INSERT INTO chat_sessions (chat_id, user_id, document_ids, module)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (chat_id, user_id, file_ids or [], category)
                    )

                cur.execute(
                    """
                    INSERT INTO chat_messages (chat_id, role, content)
                    VALUES (%s, %s, %s), (%s, %s, %s)
                    """,
                    (chat_id, "user", query, chat_id, "assistant", response)
                )
                cur.execute(
                    "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = %s",
                    (chat_id,)
                )

                cache_key = hashlib.md5(f"{cleaned_query}:{category}:{user_id}:{':'.join(file_ids or [])}".encode()).hexdigest()
                cur.execute(
                    """
                    INSERT INTO response_cache (cache_key, response, expires_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        response = EXCLUDED.response,
                        expires_at = EXCLUDED.expires_at
                    """,
                    (cache_key, Json({"response": response, "chat_id": chat_id, "sources": jsonable_encoder(search_results)}), datetime.datetime.now() + datetime.timedelta(hours=1))
                )
                conn.commit()

                logger.info(f"Chat response generated for chat_id: {chat_id}, user: {user_id}")
                return {
                    "response": response,
                    "chat_id": chat_id,
                    "sources": search_results
                }
        except Exception as e:
            logger.error(f"Chat failed for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT file_id, filename, file_type, upload_date, size, category, status FROM file_metadata WHERE user_id = %s",
                    (user_id,)
                )
                documents = [
                    {
                        "file_id": str(row[0]),
                        "filename": row[1],
                        "file_type": row[2],
                        "upload_date": row[3].isoformat(),
                        "size": row[4],
                        "category": row[5],
                        "status": row[6]
                    }
                    for row in cur.fetchall()
                ]
                logger.info(f"Retrieved {len(documents)} documents for user {user_id}")
                return {"status": "success", "documents": documents}
        except Exception as e:
            logger.error(f"Failed to list documents for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Unexpected error in list_documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.delete("/documents/{file_id}", response_model=Dict[str, str])
async def delete_document(
    file_id: str,
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            qdrant_handler.delete_by_document_id(file_id)
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM file_metadata WHERE file_id = %s AND user_id = %s",
                    (file_id, user_id)
                )
                if cur.rowcount == 0:
                    logger.warning(f"File not found for deletion: file_id={file_id}, user_id={user_id}")
                    raise HTTPException(status_code=404, detail="File not found")
                conn.commit()
            logger.info(f"Deleted document: {file_id} for user {user_id}")
            return {"status": "success", "file_id": file_id}
        except Exception as e:
            logger.error(f"Failed to delete document {file_id} for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.patch("/documents/{file_id}", response_model=Dict[str, str])
async def update_document_category(
    file_id: str,
    user_id: str,
    category: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE file_metadata SET category = %s WHERE file_id = %s AND user_id = %s",
                    (category, file_id, user_id)
                )
                if cur.rowcount == 0:
                    logger.warning(f"File not found for category update: file_id={file_id}, user_id={user_id}")
                    raise HTTPException(status_code=404, detail="File not found")
                conn.commit()
                qdrant_handler.update_metadata(file_id, {"category": category})
            logger.info(f"Updated category for document {file_id} to {category} for user {user_id}")
            return {"status": "success", "file_id": file_id, "category": category}
        except Exception as e:
            logger.error(f"Failed to update document {file_id} category for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to update document category: {str(e)}")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in update_document_category: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/prompts", response_model=Dict[str, Any])
async def create_prompt(
    category: str = Form(...),
    prompt: str = Form(...),
    user_id: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    logger.debug(f"Attempting to create/update prompt for category: {category}, user_id: {user_id}")
    try:
        if not category.strip():
            logger.error("Category cannot be empty")
            raise HTTPException(status_code=400, detail="Category cannot be empty")
        if not prompt.strip():
            logger.error("Prompt cannot be empty")
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        try:
            token_count = len(tokenizer.encode(prompt))
            logger.debug(f"Prompt token count: {token_count}")
            if token_count > 1000:
                logger.error(f"Prompt exceeds 1000 token limit: {token_count} tokens")
                raise HTTPException(status_code=400, detail=f"Prompt exceeds 1000 token limit: {token_count} tokens")
        except Exception as e:
            logger.error(f"Failed to tokenize prompt: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid prompt text: {str(e)}")

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                sql_query = """
                    INSERT INTO prompts (category, prompt, user_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT ON CONSTRAINT unique_category_user
                    DO UPDATE SET prompt = EXCLUDED.prompt, updated_at = CURRENT_TIMESTAMP
                    RETURNING id, created_at, updated_at
                """
                logger.debug(f"Executing SQL: {sql_query} with params: ({category}, [prompt], {user_id})")
                cur.execute(sql_query, (category, prompt, user_id))
                row = cur.fetchone()
                if not row:
                    logger.error(f"No rows returned from prompt insertion for category: {category}, user_id: {user_id}")
                    raise HTTPException(status_code=500, detail="Failed to save prompt: No rows affected")
                conn.commit()
                logger.info(f"Successfully saved prompt for category: {category}, user_id: {user_id}, prompt_id: {row[0]}")
                return {
                    "status": "success",
                    "prompt": {
                        "id": row[0],
                        "category": category,
                        "prompt": prompt,
                        "user_id": user_id,
                        "created_at": row[1].isoformat(),
                        "updated_at": row[2].isoformat()
                    }
                }
        except psycopg2.errors.UndefinedObject as e:
            logger.error(f"Database constraint error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database constraint error: {str(e)}")
        except psycopg2.Error as e:
            logger.error(f"Database error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_prompt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/prompts", response_model=Dict[str, Any])
async def list_prompts(
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, category, prompt, created_at, updated_at FROM prompts WHERE user_id = %s",
                    (user_id,)
                )
                prompts = [
                    {
                        "id": row[0],
                        "category": row[1],
                        "prompt": row[2],
                        "created_at": row[3].isoformat(),
                        "updated_at": row[4].isoformat(),
                        "user_id": user_id
                    }
                    for row in cur.fetchall()
                ]
                logger.info(f"Retrieved {len(prompts)} prompts for user {user_id}")
                return {"status": "success", "prompts": prompts}
        except Exception as e:
            logger.error(f"Failed to list prompts for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Unexpected error in list_prompts: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/prompts/{category}", response_model=Dict[str, Any])
async def get_prompt(
    category: str,
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, prompt, created_at, updated_at FROM prompts WHERE category = %s AND user_id = %s",
                    (category, user_id)
                )
                row = cur.fetchone()
                if not row:
                    logger.warning(f"Prompt not found for category: {category}, user_id: {user_id}")
                    raise HTTPException(status_code=404, detail="Prompt not found")
                logger.info(f"Retrieved prompt for category: {category}, user_id: {user_id}")
                return {
                    "status": "success",
                    "prompt": {
                        "id": row[0],
                        "category": category,
                        "prompt": row[1],
                        "user_id": user_id,
                        "created_at": row[2].isoformat(),
                        "updated_at": row[3].isoformat()
                    }
                }
        except Exception as e:
            logger.error(f"Failed to get prompt for category {category}, user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to get prompt: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Unexpected error in get_prompt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.delete("/prompts/{category}", response_model=Dict[str, str])
async def delete_prompt(
    category: str,
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM prompts WHERE category = %s AND user_id = %s",
                    (category, user_id)
                )
                if cur.rowcount == 0:
                    logger.warning(f"Prompt not found for deletion: category={category}, user_id={user_id}")
                    raise HTTPException(status_code=404, detail="Prompt not found")
                conn.commit()
                logger.info(f"Deleted prompt for category: {category}, user_id: {user_id}")
                return {"status": "success", "category": category}
        except Exception as e:
            logger.error(f"Failed to delete prompt for category {category}, user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete prompt: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Unexpected error in delete_prompt: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/preview/{file_id}")
async def preview_file(file_id: str, user_id: str, api_key: str = Depends(validate_api_key)):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content, file_type FROM file_metadata WHERE file_id = %s AND user_id = %s",
                    (file_id, user_id)
                )
                row = cur.fetchone()
                if not row:
                    logger.warning(f"File not found for preview: file_id={file_id}, user_id={user_id}")
                    raise HTTPException(status_code=404, detail="File not found")
                content, file_type = row
            logger.info(f"Previewing file: {file_id} for user {user_id}")
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
                BytesIO(content),
                media_type=mime_map.get(file_type, "application/octet-stream")
            )
        except Exception as e:
            logger.error(f"Failed to preview file {file_id} for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to preview file: {str(e)}")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in preview_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/chat_sessions", response_model=Dict[str, Any])
async def list_chat_sessions(
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chat_id, created_at, updated_at, document_ids, module
                    FROM chat_sessions WHERE user_id = %s
                    """,
                    (user_id,)
                )
                sessions = [
                    {
                        "chat_id": str(row[0]),
                        "user_id": user_id,
                        "created_at": row[1].isoformat(),
                        "updated_at": row[2].isoformat(),
                        "document_ids": [str(id) for id in row[3] or []],
                        "module": row[4]
                    }
                    for row in cur.fetchall()
                ]
                for session in sessions:
                    cur.execute(
                        "SELECT role, content, timestamp FROM chat_messages WHERE chat_id = %s ORDER BY timestamp",
                        (session["chat_id"],)
                    )
                    session["messages"] = [
                        {"role": row[0], "content": row[1], "timestamp": row[2].isoformat()}
                        for row in cur.fetchall()
                    ]
                logger.info(f"Retrieved {len(sessions)} chat sessions for user {user_id}")
                return {"status": "success", "chat_sessions": sessions}
        except Exception as e:
            logger.error(f"Failed to list chat sessions for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to list chat sessions: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Unexpected error in list_chat_sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.delete("/chat_sessions/{chat_id}", response_model=Dict[str, str])
async def delete_chat_session(
    chat_id: str,
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_sessions WHERE chat_id = %s AND user_id = %s",
                    (chat_id, user_id)
                )
                if cur.rowcount == 0:
                    logger.warning(f"Chat session not found for deletion: chat_id={chat_id}, user_id={user_id}")
                    raise HTTPException(status_code=404, detail="Chat session not found")
                conn.commit()
                logger.info(f"Deleted chat session: {chat_id} for user {user_id}")
                return {"status": "success", "chat_id": chat_id}
        except Exception as e:
            logger.error(f"Failed to delete chat session {chat_id} for user {user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to delete chat session: {str(e)}")
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in delete_chat_session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )