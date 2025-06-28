from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from converters import image_converter, doc_converter, excel_converter, pdf_converter, txt_converter
from utils.qdrant_handler import QdrantHandler
from utils.text_processor import TextProcessor
import os
import uuid
import datetime
import logging
import base64
import openai
from typing import List, Dict, Optional
import json
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'images': ['.heic', '.jpg', '.jpeg', '.png'],
    'documents': ['.doc', '.docx'],
    'spreadsheets': ['.xls', '.xlsx', '.csv'],
    'pdfs': ['.pdf'],
    'text': ['.txt']
}

# Initialize handlers
try:
    qdrant_handler = QdrantHandler(host="qdrant", port=6333, collection_name="rag_chunks")
except Exception as e:
    logger.error(f"Qdrant connection failed: {str(e)}")
    raise

try:
    text_processor = TextProcessor()
except Exception as e:
    logger.error(f"TextProcessor initialization failed: {str(e)}")
    text_processor = None

# State storage
STATE_FILE = "/app/data/state.json"

def save_state(file_metadata: List[Dict], chat_sessions: Dict):
    """Save file_metadata and chat_sessions to JSON"""
    state = {
        "file_metadata": file_metadata,
        "chat_sessions": chat_sessions
    }
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        logger.info("State saved to state.json")
    except Exception as e:
        logger.error(f"Failed to save state: {str(e)}")

def load_state():
    """Load file_metadata and chat_sessions from JSON"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                return state.get("file_metadata", []), state.get("chat_sessions", {})
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
    return [], {}

file_metadata, chat_sessions = load_state()

def get_file_converter(file_ext: str):
    """Return the appropriate converter for the file extension"""
    if file_ext in SUPPORTED_EXTENSIONS['images']:
        return image_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['documents']:
        return doc_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['spreadsheets']:
        return excel_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['pdfs']:
        return pdf_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['text']:
        return txt_converter.convert_to_markdown
    return None

@app.post("/process_file")
async def process_file(file: UploadFile = File(...), user_id: str = Form(...)):
    """Upload and process a file"""
    if file.size > 200 * 1024 * 1024:  # 200MB limit
        raise HTTPException(status_code=400, detail="File size exceeds 200MB limit")

    file_ext = os.path.splitext(file.filename)[1].lower()
    file_id = str(uuid.uuid4())
    output_dir = "temp_uploads"
    os.makedirs(output_dir, exist_ok=True)
    temp_path = f"{output_dir}/{file_id}{file_ext}"
    
    try:
        # Save uploaded file
        file_content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Get appropriate converter
        converter = get_file_converter(file_ext)
        if not converter:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_ext}")

        markdown_content = converter(temp_path)
        
        # Process chunks if text processor is available
        if text_processor:
            try:
                cleaned_markdown = text_processor.clean_markdown(markdown_content)
                chunks = text_processor.chunk_text(cleaned_markdown)
                chunks = text_processor.generate_embeddings(chunks)
                
                for chunk in chunks:
                    try:
                        chunk['document_id'] = file_id
                        qdrant_handler.save_chunk(chunk, user_id)
                    except Exception as e:
                        logger.error(f"Chunk save failed for {file_id}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Processing failed for {file_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Content processing failed: {str(e)}")
        
        # Store file metadata
        if not any(f['file_id'] == file_id for f in file_metadata):
            file_metadata.append({
                "file_id": file_id,
                "filename": file.filename,
                "file_type": file_ext,
                "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": base64.b64encode(file_content).decode(),
                "markdown_content": markdown_content,
                "user_id": user_id
            })
            save_state(file_metadata, chat_sessions)
            logger.info(f"Processed file: {file.filename} (ID: {file_id})")
        
        return {"status": "success", "file_id": file_id, "filename": file.filename}
    
    except Exception as e:
        logger.error(f"File processing error for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Temp file cleanup failed for {temp_path}: {str(e)}")

@app.post("/search")
async def search_documents(query: str = Form(...), user_id: str = Form(...), file_id: Optional[str] = Form(None), limit: int = Form(5)):
    """Search documents and return relevant chunks"""
    try:
        # Generate query embedding
        query_embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )['data'][0]['embedding']
        
        # Build Qdrant filter
        filters = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
        if file_id:
            filters["must"].append({"key": "document_id", "match": {"value": file_id}})
        
        # Search Qdrant
        results = qdrant_handler.client.search(
            collection_name="rag_chunks",
            query_vector=query_embedding,
            query_filter=filters,
            limit=limit
        )
        
        file_map = {f['file_id']: f['filename'] for f in file_metadata}
        
        return {
            "status": "success",
            "results": [
                {
                    "chunk_id": str(r.id),
                    "document_id": r.payload.get("document_id", "N/A"),
                    "filename": file_map.get(r.payload.get("document_id", "N/A"), "Unknown"),
                    "parent_section": r.payload.get("parent_section", "N/A"),
                    "chunk_index": r.payload.get("chunk_index", "N/A"),
                    "content": r.payload.get("content", "N/A"),
                    "score": r.score
                }
                for r in results
            ]
        }
    
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/chat")
async def chat_with_documents(query: str = Form(...), user_id: str = Form(...), file_ids: Optional[List[str]] = Form(None)):
    """Generate a chat response based on document context"""
    try:
        # Generate query embedding
        query_embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )['data'][0]['embedding']
        
        # Build Qdrant filter
        filters = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
        if file_ids:
            filters["must"].append({"key": "document_id", "match": {"any": file_ids}})
        
        # Search Qdrant
        results = qdrant_handler.client.search(
            collection_name="rag_chunks",
            query_vector=query_embedding,
            query_filter=filters,
            limit=5
        )
        
        if not results:
            return {"query": query, "response": "No relevant chunks found for the query.", "sources": []}
        
        # Prepare context
        file_map = {f['file_id']: f['filename'] for f in file_metadata}
        context = "\n\n".join([
            f"📄 **{file_map.get(r.payload.get('document_id', 'Unknown'), 'Unknown')}** "
            f"(Section {r.payload.get('chunk_index', 'N/A')}):\n"
            f"{r.payload.get('content', '')}\n"
            for r in results
        ])
        
        # Generate response
        prompt = (
            f"Context:\n{context}\n\n"
            f"Query: {query}\n\n"
            "Answer the query based on the provided context. "
            "Cite the document name and section where information is sourced."
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers queries based on document context, citing sources clearly."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        answer = response['choices'][0]['message']['content']
        
        # Store in chat history
        chat_id = str(uuid.uuid4())
        if chat_id not in chat_sessions:
            chat_sessions[chat_id] = []
        chat_sessions[chat_id].append({
            "query": query,
            "response": answer,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_state(file_metadata, chat_sessions)
        
        return {
            "query": query,
            "response": answer,
            "chat_id": chat_id,
            "sources": [
                {
                    "filename": file_map.get(r.payload.get("document_id", "N/A"), "Unknown"),
                    "chunk_index": r.payload.get("chunk_index", "N/A"),
                    "score": r.score
                }
                for r in results
            ]
        }
    
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/documents")
async def list_documents(user_id: str):
    """List all documents for a user"""
    try:
        user_docs = [f for f in file_metadata if f.get("user_id") == user_id]
        return {
            "status": "success",
            "documents": [
                {
                    "file_id": f["file_id"],
                    "filename": f["filename"],
                    "file_type": f["file_type"],
                    "upload_date": f["upload_date"]
                }
                for f in user_docs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{file_id}")
async def delete_document(file_id: str, user_id: str):
    """Delete a document and its chunks"""
    try:
        qdrant_handler.delete_by_document_id(file_id)
        global file_metadata
        file_metadata = [f for f in file_metadata if f["file_id"] != file_id]
        save_state(file_metadata, chat_sessions)
        logger.info(f"Deleted document: {file_id}")
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        logger.error(f"Failed to delete document {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")