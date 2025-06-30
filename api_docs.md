# pdfLLM API Documentation

The `pdfLLM` FastAPI backend (`app/main.py`) provides RESTful endpoints for programmatic access to document processing, search, chat, and management functionalities. All endpoints require an `X-API-Key` header with a valid OpenAI API key and a `user_id` to scope data to specific users. The API integrates with Qdrant for vector storage and OpenAI for embeddings and chat responses.

## Endpoints

### 1. `POST /process_file`
**Description**: Upload and process a file, converting it to markdown, chunking content, generating OpenAI embeddings, and storing chunks in Qdrant. Metadata is saved in `state.json`.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (UploadFile): File to process (e.g., `.pdf`, `.txt`, `.docx`, `.xlsx`, `.png`). Max size: 200MB.
  - `user_id` (str): Unique user identifier.
- **Headers**:
  - `X-API-Key`: OpenAI API key.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/process_file   -H "X-API-Key: your-openai-api-key"   -F "file=@example_document.pdf"   -F "user_id=default_user"
  ```

**Response**:
- **Status**: 200 OK
- **Body**:
  ```json
  {
    "status": "success",
    "file_id": "aae5b99b-8145-4259-b4f6-f46aee4e67bd",
    "filename": "example_document.pdf"
  }
  ```

### 2. `POST /search`
**Description**: Search Qdrant for document chunks relevant to a query using OpenAI embeddings and entity-based filtering.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (str): Search query.
  - `user_id` (str): Unique user identifier.
  - `file_id` (str, optional): Filter by document ID.
  - `limit` (int, default=5): Maximum chunks to return.
  - `use_graph` (bool, default=True): Include knowledge graph-based retrieval.
- **Headers**:
  - `X-API-Key`: OpenAI API key.

**Example**:
```bash
curl -X POST http://localhost:8000/search -H "X-API-Key: your-openai-api-key" -F "query=What is the requisition amount for the project?" -F "user_id=default_user" -F "file_id=aae5b99b-8145-4259-b4f6-f46aee4e67bd"
```

**Response**:
```json
{
  "status": "success",
  "results": [
    {
      "chunk_id": "uuid-string",
      "document_id": "aae5b99b-8145-4259-b4f6-f46aee4e67bd",
      "filename": "example_document.pdf",
      "parent_section": "Section 1",
      "chunk_index": 1,
      "content": "CURRENT PAYMENT DUE: $29,825.80",
      "entities": ["Example Contractor Inc."],
      "relationships": [{"subject": "Example Contractor Inc.", "predicate": "appears_in", "object": "aae5b99b-8145-4259-b4f6-f46aee4e67bd"}],
      "score": 0.95
    }
  ]
}
```

### 3. `POST /chat`
**Description**: Generate a chat response using OpenAI’s model with context from Qdrant-retrieved chunks.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (str): User’s question.
  - `user_id` (str): Unique user identifier.
  - `file_ids` (List[str], optional): List of document IDs for context.
  - `chat_id` (str, optional): Existing chat session ID.
- **Headers**:
  - `X-API-Key`: OpenAI API key.

**Example**:
```bash
curl -X POST http://localhost:8000/chat -H "X-API-Key: your-openai-api-key" -F "query=What is the upcoming requisition amount for the project?" -F "user_id=default_user" -F "file_ids=aae5b99b-8145-4259-b4f6-f46aee4e67bd"
```

**Response**:
```json
{
  "response": "The requisition amount is $29,825.80, noted in Section 1 of example_document.pdf under 'CURRENT PAYMENT DUE'.",
  "chat_id": "uuid-string",
  "sources": [
    {
      "chunk_id": "uuid-string",
      "document_id": "aae5b99b-8145-4259-b4f6-f46aee4e67bd",
      "filename": "example_document.pdf",
      "parent_section": "Section 1",
      "chunk_index": 1,
      "content": "CURRENT PAYMENT DUE: $29,825.80",
      "entities": ["Example Contractor Inc."],
      "relationships": [{"subject": "Example Contractor Inc.", "predicate": "appears_in", "object": "aae5b99b-8145-4259-b4f6-f46aee4e67bd"}],
      "score": 0.95
    }
  ]
}
```

### 4. `GET /documents`
**Description**: List all documents uploaded by a user.

**Request**:
- **Parameters**:
  - `user_id` (str): Unique user identifier.
- **Headers**:
  - `X-API-Key`: OpenAI API key.

### 5. `DELETE /documents/{file_id}`
**Description**: Delete a document and its Qdrant chunks.

**Request**:
- **Parameters**:
  - `file_id` (str): Document ID to delete.
  - `user_id` (str): Unique user identifier.
- **Headers**:
  - `X-API-Key`: OpenAI API key.

### 6. `GET /preview/{file_id}`
**Description**: Stream a file’s content for preview.

**Request**:
- **Parameters**:
  - `file_id` (str): Document ID to preview.
  - `user_id` (str): Unique user identifier.
- **Headers**:
  - `X-API-Key`: OpenAI API key.

### 7. `GET /health`
**Description**: Check the health of the FastAPI service.

**Response**:
```json
{
  "status": "healthy"
}
```