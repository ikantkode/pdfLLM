# pdfLLM API Documentation

The `pdfLLM` FastAPI backend provides RESTful endpoints to upload, search, chat with, and manage documents. This API combines semantic and graph-based retrieval to deliver intelligent answers from document context.

> 🛡 All endpoints require:
> - `X-API-Key` header
> - `user_id` (as form/query param)

---

## 🔄 `POST /process_file`

**Description**: Upload and convert a document. Generates embeddings and stores metadata + vectors.

**Form Fields**:
- `file`: The document to upload.
- `user_id`: Your user/session ID.

**Headers**:
- `X-API-Key`: Your API key.

**Response**:
```json
{
  "status": "success",
  "file_id": "uuid-string",
  "filename": "your_file.pdf"
}
```

---

## 🔍 `POST /search`

**Description**: Search documents using hybrid retrieval (semantic + knowledge graph).

**Form Fields**:
- `query`: The user question.
- `user_id`: User/session ID.
- `file_ids`: Optional list of file UUIDs.
- `limit`: Max number of results (default: 5).
- `use_graph`: Boolean for using knowledge graph (default: true).

**Response**:
```json
{
  "status": "success",
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "doc_uuid",
      "filename": "your_file.pdf",
      "parent_section": "Section 0",
      "chunk_index": 0,
      "content": "text from the file...",
      "entities": ["Entity A"],
      "relationships": [{"subject": "Entity A", "predicate": "appears_in", "object": "doc_uuid"}],
      "score": 0.93
    }
  ]
}
```

---

## 💬 `POST /chat`

**Description**: Ask questions about uploaded documents. Returns structured and formatted answers.

**Form Fields**:
- `query`: Your question.
- `user_id`: Your user/session ID.
- `file_ids`: Optional list of document UUIDs.
- `chat_id`: Optional chat session ID (for continuity).

**Response**:
```json
{
  "response": "Here is the answer to your query...",
  "chat_id": "uuid",
  "sources": [/* matching chunks */]
}
```

---

## 📂 `GET /documents`

**Description**: List all uploaded files for a user.

**Query Parameters**:
- `user_id`: Your user/session ID.

**Headers**:
- `X-API-Key`: Your API key.

**Response**:
```json
{
  "status": "success",
  "documents": [
    {
      "file_id": "uuid",
      "filename": "doc.pdf",
      "file_type": ".pdf",
      "upload_date": "2025-07-09T14:12:33",
      "size": 302581
    }
  ]
}
```

---

## 🗑 `DELETE /documents/{file_id}`

**Description**: Delete a document and its stored embeddings.

**Query Parameters**:
- `user_id`: Your user/session ID.

**Path Parameter**:
- `file_id`: UUID of the document.

---

## 👁 `GET /preview/{file_id}`

**Description**: Preview the raw content of an uploaded document.

**Query Parameters**:
- `user_id`: Your user/session ID.

**Response**: `StreamingResponse` of the decoded file content.

---

## 🧠 `GET /knowledge_graph`

**Description**: Get a graph structure of all entities and relationships.

**Query Parameters**:
- `user_id`: Your user/session ID.
- `file_id` (optional): Filter for one document.

**Response**:
```json
{
  "status": "success",
  "nodes": [{"id": "entity", "label": "entity", "type": "entity"}],
  "edges": [{"from": "entity1", "to": "entity2", "label": "appears_in", "weight": 1.0}]
}
```

---

## ❤️ `GET /health`

**Description**: Check if the server is up and running.

**Response**:
```json
{ "status": "healthy" }
```
