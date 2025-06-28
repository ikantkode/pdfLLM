# pdfLLM API Documentation

The `pdfLLM` API is a FastAPI-based RESTful interface for a Retrieval-Augmented Generation (RAG) microservice. It enables users to upload documents, process them into markdown, store text chunks in a Qdrant vector database with OpenAI embeddings, and query or chat with the content. The API supports multiple file formats (`.pdf`, `.txt`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.csv`, `.jpg`, `.jpeg`, `.png`, `.heic`) and provides endpoints for file processing, searching, chatting, listing documents, and deleting documents. All endpoints require a `user_id` parameter to scope data to specific users, stored in `state.json` for persistence.

## Base URL
- **URL**: `http://localhost:8000`
- **Deployment**: Runs via Docker Compose (`rag-service` in `docker-compose.yml`), accessible after starting the application.

## Authentication
- No explicit authentication is required.
- The `user_id` (string) parameter in each endpoint scopes data to a specific user, ensuring isolation of documents and chat sessions.

## Endpoints

### 1. Upload and Process File
**`POST /process_file`**

**Description**: Uploads a file, converts it to markdown, chunks the content, generates OpenAI embeddings, and stores chunks in Qdrant. Metadata is saved in `state.json`.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (UploadFile, required): The file to process. Supported formats: `.pdf`, `.txt`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.csv`, `.jpg`, `.jpeg`, `.png`, `.heic`. Max size: 200MB.
  - `user_id` (string, required): Unique identifier for the user.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/process_file \
  -F "file=@notes.txt" \
  -F "user_id=test_user"
  ```

**Response**:
- **Status**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "status": "success",
    "file_id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "notes.txt"
  }
  ```
- **Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "status": { "type": "string", "enum": ["success"] },
      "file_id": { "type": "string", "format": "uuid" },
      "filename": { "type": "string" }
    },
    "required": ["status", "file_id", "filename"]
  }
  ```
- **Errors**:
  - `400 Bad Request`: File size exceeds 200MB or unsupported file format.
    ```json
    {
      "detail": "File size exceeds 200MB limit"
    }
    ```
    ```json
    {
      "detail": "Unsupported file format: .xyz"
    }
    ```
  - `500 Internal Server Error`: Processing or Qdrant save failure.
    ```json
    {
      "detail": "Failed to process file: [error message]"
    }
    ```

### 2. Search Documents
**`POST /search`**

**Description**: Searches for relevant document chunks in Qdrant using an OpenAI embedding of the query. Optionally filters by a specific document.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (string, required): The search query.
  - `user_id` (string, required): Unique identifier for the user.
  - `file_id` (string, optional): Filter results to a specific document ID.
  - `limit` (integer, optional, default=5): Maximum number of chunks to return.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/search \
  -F "query=What is this document about?" \
  -F "user_id=test_user" \
  -F "file_id=550e8400-e29b-41d4-a716-446655440000"
  ```

**Response**:
- **Status**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "status": "success",
    "results": [
      {
        "chunk_id": "123e4567-e89b-12d3-a456-426614174000",
        "document_id": "550e8400-e29b-41d4-a716-446655440000",
        "filename": "notes.txt",
        "parent_section": "Introduction",
        "chunk_index": 1,
        "content": "This document discusses the project overview...",
        "score": 0.95
      }
    ]
  }
  ```
- **Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "status": { "type": "string", "enum": ["success"] },
      "results": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "chunk_id": { "type": "string", "format": "uuid" },
            "document_id": { "type": "string", "format": "uuid" },
            "filename": { "type": "string" },
            "parent_section": { "type": "string" },
            "chunk_index": { "type": "integer" },
            "content": { "type": "string" },
            "score": { "type": "number" }
          },
          "required": ["chunk_id", "document_id", "filename", "parent_section", "chunk_index", "content", "score"]
        }
      }
    },
    "required": ["status", "results"]
  }
  ```
- **Errors**:
  - `500 Internal Server Error`: Qdrant or embedding generation failure.
    ```json
    {
      "detail": "Search failed: [error message]"
    }
    ```

### 3. Chat with Documents
**`POST /chat`**

**Description**: Generates a chat response using OpenAI’s `gpt-4o-mini` model, based on relevant document chunks retrieved from Qdrant. Optionally filters by specific documents.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (string, required): The user’s question.
  - `user_id` (string, required): Unique identifier for the user.
  - `file_ids` (array of strings, optional): List of document IDs to filter context.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/chat \
  -F "query=What was discussed in the meeting?" \
  -F "user_id=test_user" \
  -F "file_ids=550e8400-e29b-41d4-a716-446655440000"
  ```

**Response**:
- **Status**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "query": "What was discussed in the meeting?",
    "response": "The meeting discussed project goals... (sourced from notes.txt, Section 1)",
    "chat_id": "789a0123-b456-789c-d012-345678901234",
    "sources": [
      {
        "filename": "notes.txt",
        "chunk_index": 1,
        "score": 0.95
      }
    ]
  }
  ```
- **Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "query": { "type": "string" },
      "response": { "type": "string" },
      "chat_id": { "type": "string", "format": "uuid" },
      "sources": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "filename": { "type": "string" },
            "chunk_index": { "type": "integer" },
            "score": { "type": "number" }
          },
          "required": ["filename", "chunk_index", "score"]
        }
      }
    },
    "required": ["query", "response", "chat_id", "sources"]
  }
  ```
- **Errors**:
  - `500 Internal Server Error`: Qdrant or OpenAI failure.
    ```json
    {
      "detail": "Chat failed: [error message]"
    }
    ```

### 4. List Documents
**`GET /documents`**

**Description**: Retrieves a list of all documents uploaded by a user, stored in `state.json`.

**Request**:
- **Parameters** (query):
  - `user_id` (string, required): Unique identifier for the user.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/documents?user_id=test_user"
  ```

**Response**:
- **Status**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "status": "success",
    "documents": [
      {
        "file_id": "550e8400-e29b-41d4-a716-446655440000",
        "filename": "notes.txt",
        "file_type": ".txt",
        "upload_date": "2025-06-27 20:30:00"
      }
    ]
  }
  ```
- **Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "status": { "type": "string", "enum": ["success"] },
      "documents": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "file_id": { "type": "string", "format": "uuid" },
            "filename": { "type": "string" },
            "file_type": { "type": "string" },
            "upload_date": { "type": "string", "format": "date-time" }
          },
          "required": ["file_id", "filename", "file_type", "upload_date"]
        }
      }
    },
    "required": ["status", "documents"]
  }
  ```
- **Errors**:
  - `500 Internal Server Error`: State file access failure.
    ```json
    {
      "detail": "Failed to list documents: [error message]"
    }
    ```

### 5. Delete Document
**`DELETE /documents/{file_id}`**

**Description**: Deletes a document and its associated Qdrant chunks, updating `state.json`.

**Request**:
- **Parameters**:
  - `file_id` (string, path, required): The document ID to delete.
  - `user_id` (string, query, required): Unique identifier for the user.
- **Example**:
  ```bash
  curl -X DELETE "http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000?user_id=test_user"
  ```

**Response**:
- **Status**: `200 OK`
- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "status": "success",
    "file_id": "550e8400-e29b-41d4-a716-446655440000"
  }
  ```
- **Schema**:
  ```json
  {
    "type": "object",
    "properties": {
      "status": { "type": "string", "enum": ["success"] },
      "file_id": { "type": "string", "format": "uuid" }
    },
    "required": ["status", "file_id"]
  }
  ```
- **Errors**:
  - `500 Internal Server Error`: Qdrant or state file failure.
    ```json
    {
      "detail": "Failed to delete document: [error message]"
    }
    ```

## Notes
- **File Size Limit**: Uploads are capped at 200MB to prevent resource exhaustion.
- **Supported Formats**: `.pdf`, `.txt`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.csv`, `.jpg`, `.jpeg`, `.png`, `.heic`.
- **State Persistence**: `file_metadata` and `chat_sessions` are stored in `/app/data/state.json`, shared between FastAPI and Streamlit services via a Docker volume.
- **OpenAI Models**: Uses `text-embedding-3-small` for embeddings and `gpt-4o-mini` for chat responses. Upgrade to `openai==1.40.0` for async support if needed.
- **Qdrant**: Stores chunks in the `rag_chunks` collection, accessible at `http://localhost:6333` for debugging.
- **Testing**: Use the Streamlit frontend (`http://localhost:8501`) to interact with the API and debug Qdrant chunks.

## Example Workflow
1. **Upload a File**:
   ```bash
   curl -X POST http://localhost:8000/process_file \
   -F "file=@document.pdf" \
   -F "user_id=test_user"
   ```
2. **List Documents**:
   ```bash
   curl -X GET "http://localhost:8000/documents?user_id=test_user"
   ```
3. **Search for Content**:
   ```bash
   curl -X POST http://localhost:8000/search \
   -F "query=What is the document about?" \
   -F "user_id=test_user"
   ```
4. **Chat with Documents**:
   ```bash
   curl -X POST http://localhost:8000/chat \
   -F "query=What was discussed in the meeting?" \
   -F "user_id=test_user" \
   -F "file_ids=550e8400-e29b-41d4-a716-446655440000"
   ```
5. **Delete a Document**:
   ```bash
   curl -X DELETE "http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000?user_id=test_user"
   ```

## Troubleshooting
- **API Not Responding**:
  - Check container status:
    ```bash
    docker ps
    docker logs pdfllm-rag-service-1
    ```
  - Ensure `OPENAI_API_KEY` is set in `.env`.
- **State Issues**:
  - Verify `state.json`:
    ```bash
    cat app/data/state.json
    chmod -R 777 app/data
    ```
- **Qdrant Errors**:
  - Check Qdrant logs:
    ```bash
    docker logs pdfllm-qdrant-1
    ```
  - Verify collection:
    ```bash
    docker exec -it pdfllm-qdrant-1 curl -s http://localhost:6333/collections/rag_chunks
    ```
- **Generic Headings**: If chunks show `# PDF Content` in `parent_section`, update `app/converters/pdf_converter.py` to extract meaningful headings.