# API Documentation

## Overview
This is a Retrieval-Augmented Generation (RAG) microservice built with FastAPI. It supports file uploading and processing (with OCR), document querying via chat, prompt management, and session handling. Authentication uses an API key header (X-API-Key). The service is available at /api/docs (Swagger) and /api/redoc (ReDoc) for interactive docs.

All endpoints require the X-API-Key header for validation. Responses are JSON unless specified otherwise. Errors return HTTP status codes with {"detail": "message"}.

## Endpoints

### POST /process_file
**Description**: Upload and queue a file for processing (OCR, embedding, storage in Qdrant).  
**Parameters** (multipart/form-data):  
- file: UploadFile (required) - The file to process.  
- user_id: str (required) - User identifier.  
- category: str (optional) - Document category (e.g., "payrolls").  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "file_id": str, "filename": str}.  
**Errors**: 400 (file too large or invalid config), 401 (invalid key), 500 (processing error).

### POST /chat
**Description**: Query documents with a natural language question, retrieving and generating a response using RAG.  
**Parameters** (multipart/form-data):  
- query: str (required) - The user query.  
- user_id: str (required) - User identifier.  
- file_ids: List[str] (optional) - Specific document IDs to query.  
- chat_id: str (optional) - Existing chat session ID.  
- category: str (optional, default="all") - Filter by category.  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"response": str, "chat_id": str, "sources": List[SearchResult]}.  
**Errors**: 401 (invalid key), 500 (generation error).

### GET /documents
**Description**: List uploaded documents for a user.  
**Query Parameters**:  
- user_id: str (required) - User identifier.  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "documents": List[Dict] (with file_id, filename, etc.)}.  
**Errors**: 401 (invalid key), 500 (DB error).

### DELETE /documents/{file_id}
**Description**: Delete a document and its chunks from Qdrant.  
**Path Parameters**: file_id: str (required).  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "file_id": str}.  
**Errors**: 401 (invalid key), 404 (not found), 500 (deletion error).

### PATCH /documents/{file_id}
**Description**: Update a document's category.  
**Path Parameters**: file_id: str (required).  
**Parameters** (multipart/form-data):  
- category: str (required) - New category.  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "file_id": str, "category": str}.  
**Errors**: 401 (invalid key), 404 (not found), 500 (update error).

### POST /prompts
**Description**: Create or update a system prompt for a category.  
**Parameters** (multipart/form-data):  
- category: str (required) - Prompt category.  
- prompt: str (required) - Prompt text.  
- user_id: str (required) - User identifier.  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "prompt": Dict (with id, category, etc.)}.  
**Errors**: 400 (prompt too long), 401 (invalid key), 500 (DB error).

### GET /prompts
**Description**: List all prompts for a user.  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "prompts": List[Dict]}.  
**Errors**: 401 (invalid key), 500 (DB error).

### GET /prompts/{category}
**Description**: Get a specific prompt by category.  
**Path Parameters**: category: str (required).  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "prompt": Dict}.  
**Errors**: 401 (invalid key), 404 (not found), 500 (DB error).

### DELETE /prompts/{category}
**Description**: Delete a prompt by category.  
**Path Parameters**: category: str (required).  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "category": str}.  
**Errors**: 401 (invalid key), 404 (not found), 500 (DB error).

### GET /preview/{file_id}
**Description**: Stream a preview of the file content.  
**Path Parameters**: file_id: str (required).  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - StreamingResponse (file bytes, with appropriate MIME type).  
**Errors**: 401 (invalid key), 404 (not found), 500 (DB error).

### GET /chat_sessions
**Description**: List all chat sessions for a user, including messages.  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "chat_sessions": List[Dict] (with chat_id, messages, etc.)}.  
**Errors**: 401 (invalid key), 500 (DB error).

### DELETE /chat_sessions/{chat_id}
**Description**: Delete a chat session.  
**Path Parameters**: chat_id: str (required).  
**Query Parameters**: user_id: str (required).  
**Headers**: X-API-Key (required).  
**Response**: 200 OK - {"status": "success", "chat_id": str}.  
**Errors**: 401 (invalid key), 404 (not found), 500 (DB error).

## Models
- **FileMetadata**: file_id, filename, file_type, upload_date, content (optional), markdown_content (optional), user_id, size, checksum, category (optional), status.  
- **ChatMessage**: message_id, chat_id, role, content, timestamp.  
- **ChatSession**: chat_id, user_id, created_at, updated_at, document_ids, module (optional).  
- **SearchResult**: chunk_id, document_id, filename, parent_section, chunk_index, content, entities, relationships, score, category (optional).  
- **Prompt**: id, category, prompt, created_at, updated_at, user_id.

## Security
- All endpoints require X-API-Key matching the configured key (for OpenAI-enabled mode).  
- User isolation via user_id in queries and filters.

## Notes
- File processing is asynchronous via Celery. Check status via GET /documents.  
- Responses may be cached for 1 hour based on query hash.  
- Supported file types: images (jpg, png, etc.), documents (docx, pdf), spreadsheets (xlsx), text (txt).