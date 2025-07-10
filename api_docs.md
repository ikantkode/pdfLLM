
# pdfLLM API Documentation

The `pdfLLM` FastAPI backend provides RESTful endpoints to upload, search, chat with, and manage documents. This API combines semantic and graph-based retrieval to deliver intelligent answers from document context.

> 🛡 All endpoints require:
> - `X-API-Key` header (e.g., `X-API-Key: sk-sample1234567890`)
> - `user_id` (as form/query param)

---

## 🔄 `POST /process_file`

**Description**: Upload and convert a document. Generates embeddings and stores metadata + vectors.

**Headers**:
- `X-API-Key`: Your API key
- `Content-Type`: `multipart/form-data`

**Form Fields**:
- `file`: (required) The document to upload
- `user_id`: (required) Your user/session ID

**Example Request**:
```bash
curl -X POST "http://localhost:8000/process_file" \
  -H "X-API-Key: sk-sample1234567890" \
  -F "file=@document.pdf" \
  -F "user_id=test_user"
```

**Success Response**:
```json
{
  "status": "success",
  "file_id": "c4c82b21-469f-403f-bed7-87ce5b167a8d",
  "filename": "document.pdf"
}
```

## 🔍 `POST /search`

**Description**: Search documents using hybrid retrieval (semantic + knowledge graph).

**Headers**:
- `X-API-Key`: Your API key
- `Content-Type`: `application/x-www-form-urlencoded`

**Form Fields**:
- `query`: (required) The search query
- `user_id`: (required) User/session ID
- `file_ids`: (optional) Comma-separated list of file UUIDs
- `limit`: (optional) Max results (default: 5)
- `use_graph`: (optional) Boolean for knowledge graph (default: true)

**Example Request**:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "X-API-Key: sk-sample1234567890" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=insurance coverage&user_id=test_user&limit=3"
```

**Response**:
```json
{
  "status": "success",
  "results": [
    {
      "chunk_id": "022f5020-f6fe-4177-9d03-f50debd93939",
      "document_id": "c4c82b21-469f-403f-bed7-87ce5b167a8d",
      "filename": "insurance.pdf",
      "parent_section": "Section 2",
      "content": "The policy covers general liability up to $10M...",
      "score": 0.92
    }
  ]
}
```

## 💬 `POST /chat`

**Description**: Ask questions about uploaded documents with contextual understanding.

**Headers**:
- `X-API-Key`: Your API key
- `Content-Type`: `application/x-www-form-urlencoded`

**Form Fields**:
- `query`: (required) Your question
- `user_id`: (required) User/session ID
- `file_ids`: (optional) Comma-separated document UUIDs
- `chat_id`: (optional) Existing chat session ID

**Example Request**:
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "X-API-Key: sk-sample1234567890" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=What's the coverage limit?&user_id=test_user"
```

**Response**:
```json
{
  "response": "The general liability coverage has a $10M limit per occurrence...",
  "chat_id": "bd20d3c7-8c1e-409e-bef2-dbf1ec1feb57",
  "sources": [
    {
      "chunk_id": "022f5020-f6fe-4177-9d03-f50debd93939",
      "document_id": "c4c82b21-469f-403f-bed7-87ce5b167a8d",
      "filename": "insurance.pdf",
      "content": "General Liability: $10M per occurrence..."
    }
  ]
}
```

## 📂 `GET /documents`

**Description**: List all uploaded files for a user.

**Headers**:
- `X-API-Key`: Your API key

**Query Parameters**:
- `user_id`: (required) Your user/session ID

**Example Request**:
```bash
curl -X GET "http://localhost:8000/documents?user_id=test_user" \
  -H "X-API-Key: sk-sample1234567890"
```

**Response**:
```json
{
  "status": "success",
  "documents": [
    {
      "file_id": "c4c82b21-469f-403f-bed7-87ce5b167a8d",
      "filename": "insurance.pdf",
      "file_type": "application/pdf",
      "upload_date": "2025-07-10T14:30:00",
      "size": 102400
    }
  ]
}
```

## 🗑 `DELETE /documents/{file_id}`

**Description**: Delete a document and its embeddings.

**Headers**:
- `X-API-Key`: Your API key

**Path Parameters**:
- `file_id`: (required) Document UUID

**Query Parameters**:
- `user_id`: (required) Your user/session ID

**Example Request**:
```bash
curl -X DELETE "http://localhost:8000/documents/c4c82b21-469f-403f-bed7-87ce5b167a8d?user_id=test_user" \
  -H "X-API-Key: sk-sample1234567890"
```

**Success Response**:
```json
{
  "status": "success",
  "message": "Document deleted"
}
```

## 👁 `GET /preview/{file_id}`

**Description**: Preview raw document content.

**Headers**:
- `X-API-Key`: Your API key

**Query Parameters**:
- `user_id`: (required) Your user/session ID

**Response**: Raw file content

## 🧠 `GET /knowledge_graph`

**Description**: Get entity-relationship graph structure.

**Headers**:
- `X-API-Key`: Your API key

**Query Parameters**:
- `user_id`: (required) Your user/session ID
- `file_id`: (optional) Filter by document UUID

**Example Response**:
```json
{
  "status": "success",
  "nodes": [
    {"id": "ACE Insurance", "label": "Insurance Provider", "type": "organization"}
  ],
  "edges": [
    {"from": "MCT Inc", "to": "ACE Insurance", "label": "insured_by", "weight": 0.95}
  ]
}
```

## ❤️ `GET /health`

**Description**: Service health check.

**Example Request**:
```bash
curl -X GET "http://localhost:8000/health"
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "12:34:56"
}
```

## Error Responses

**401 Unauthorized**:
```json
{
  "detail": "Invalid API Key"
}
```

**404 Not Found**:
```json
{
  "detail": "Document not found"
}
```

**500 Server Error**:
```json
{
  "detail": "Internal server error"
}
```

**Note**: Replace `localhost:8000` with your actual server URL and `sk-sample1234567890` with your valid API key.
