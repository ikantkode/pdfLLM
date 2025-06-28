# pdfLLM

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice that allows users to upload, process, and query documents (e.g., PDFs, text files, Word documents, spreadsheets, images) using a FastAPI backend and a Streamlit frontend for debugging. It converts documents to markdown, stores text chunks in a Qdrant vector database, and uses OpenAI embeddings and chat models to answer queries based on document content. The application supports multiple file formats and provides a RESTful API for programmatic access and a web interface for interactive testing.

## Features
- **Document Processing**: Upload and convert files (`.pdf`, `.txt`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.csv`, `.jpg`, `.jpeg`, `.png`, `.heic`) to markdown.
- **Vector Storage**: Store document chunks in Qdrant with OpenAI embeddings for efficient retrieval.
- **Querying**: Search documents or generate chat responses using OpenAI's `gpt-4o-mini` model with context from relevant chunks.
- **FastAPI Backend**: Exposes endpoints for file processing, searching, chatting, listing, and deleting documents.
- **Streamlit Frontend**: Provides a UI for uploading files, managing documents, chatting, and debugging Qdrant chunks.
- **State Persistence**: Shares `file_metadata` and `chat_sessions` between services via `state.json`.

## Prerequisites
- **Docker**: Install [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/).
- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys).
- **System Requirements**: 4GB RAM, 10GB disk space for containers and data.
- **Supported OS**: Linux, macOS, or Windows with WSL2.

## Project Structure
```
pdfLLM/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── app/
│   ├── converters/
│   │   ├── __init__.py
│   │   ├── doc_converter.py
│   │   ├── excel_converter.py
│   │   ├── image_converter.py
│   │   ├── pdf_converter.py
│   │   ├── txt_converter.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── qdrant_handler.py
│   │   ├── text_processor.py
│   ├── data/
│   │   ├── state.json (generated at runtime)
│   ├── temp_uploads/ (generated at runtime)
│   ├── main.py (FastAPI backend)
│   ├── streamlit_app.py (Streamlit frontend)
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd pdfLLM
   ```

2. **Create `.env` File**:
   Create a `.env` file in the project root with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your-openai-api-key" > .env
   ```

3. **Create Data Directory**:
   Ensure the `app/data` directory exists for `state.json`:
   ```bash
   mkdir -p app/data
   chmod -R 777 app/data
   ```

## Deployment
Deploy the application using Docker Compose, which runs three services:
- **rag-service**: FastAPI backend on `http://localhost:8000`.
- **streamlit-service**: Streamlit frontend on `http://localhost:8501`.
- **qdrant**: Qdrant vector database on `http://localhost:6333`.

1. **Build and Start Containers**:
   ```bash
   docker-compose up --build
   ```
   - This builds the Docker image, starts the services, and maps ports `8000` (FastAPI), `8501` (Streamlit), and `6333` (Qdrant).

2. **Verify Services**:
   ```bash
   docker ps
   ```
   - Ensure `pdfllm-rag-service-1`, `pdfllm-streamlit-service-1`, and `pdfllm-qdrant-1` are `Up`.

3. **Access the Application**:
   - **FastAPI**: Test endpoints at `http://localhost:8000` (see FastAPI Endpoints below).
   - **Streamlit**: Open `http://localhost:8501` in a browser for the web interface.
   - **Qdrant**: Access the REST API at `http://localhost:6333` (optional, for debugging).

4. **Stop Containers**:
   ```bash
   docker-compose down
   ```

## FastAPI Endpoints
The FastAPI backend (`app/main.py`) provides the following endpoints for programmatic access. All endpoints require a `user_id` to scope data to specific users.

### 1. `POST /process_file`
**Description**: Upload and process a file, converting it to markdown, chunking the content, generating embeddings, and storing chunks in Qdrant. The file metadata is saved in `state.json`.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (UploadFile): The file to process (e.g., `.pdf`, `.txt`, `.docx`, `.xlsx`, `.png`). Max size: 200MB.
  - `user_id` (str): Unique identifier for the user.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/process_file \
  -F "file=@notes.txt" \
  -F "user_id=test_user"
  ```

**Response**:
- **Status**: 200 OK
- **Body**:
  ```json
  {
    "status": "success",
    "file_id": "uuid-string",
    "filename": "notes.txt"
  }
  ```
- **Errors**:
  - 400: File size exceeds 200MB or unsupported format.
  - 500: Processing or Qdrant save failure.

### 2. `POST /search`
**Description**: Search for relevant document chunks in Qdrant based on a query, using OpenAI embeddings. Optionally filter by a specific file.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (str): The search query.
  - `user_id` (str): Unique identifier for the user.
  - `file_id` (str, optional): Filter results to a specific document.
  - `limit` (int, default=5): Maximum number of chunks to return.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/search \
  -F "query=What is this document about?" \
  -F "user_id=test_user" \
  -F "file_id=uuid-string"
  ```

**Response**:
- **Status**: 200 OK
- **Body**:
  ```json
  {
    "status": "success",
    "results": [
      {
        "chunk_id": "uuid-string",
        "document_id": "uuid-string",
        "filename": "notes.txt",
        "parent_section": "Section Title",
        "chunk_index": 1,
        "content": "Chunk content...",
        "score": 0.95
      },
      ...
    ]
  }
  ```
- **Errors**: 500 (Qdrant or embedding failure).

### 3. `POST /chat`
**Description**: Generate a chat response based on relevant document chunks retrieved from Qdrant, using OpenAI's `gpt-4o-mini` model. Optionally filter by specific files.

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `query` (str): The user’s question.
  - `user_id` (str): Unique identifier for the user.
  - `file_ids` (List[str], optional): List of document IDs to filter context.
- **Example**:
  ```bash
  curl -X POST http://localhost:8000/chat \
  -F "query=What was discussed in the meeting?" \
  -F "user_id=test_user" \
  -F "file_ids=uuid-string1" \
  -F "file_ids=uuid-string2"
  ```

**Response**:
- **Status**: 200 OK
- **Body**:
  ```json
  {
    "query": "What was discussed in the meeting?",
    "response": "The meeting discussed... (sourced from notes.txt, Section 1)",
    "chat_id": "uuid-string",
    "sources": [
      {
        "filename": "notes.txt",
        "chunk_index": 1,
        "score": 0.95
      },
      ...
    ]
  }
  ```
- **Errors**: 500 (Qdrant or OpenAI failure).

### 4. `GET /documents`
**Description**: List all documents uploaded by a user, retrieved from `state.json`.

**Request**:
- **Parameters** (query):
  - `user_id` (str): Unique identifier for the user.
- **Example**:
  ```bash
  curl -X GET "http://localhost:8000/documents?user_id=test_user"
  ```

**Response**:
- **Status**: 200 OK
- **Body**:
  ```json
  {
    "status": "success",
    "documents": [
      {
        "file_id": "uuid-string",
        "filename": "notes.txt",
        "file_type": ".txt",
        "upload_date": "2025-06-27 20:30:00"
      },
      ...
    ]
  }
  ```
- **Errors**: 500 (state file access failure).

### 5. `DELETE /documents/{file_id}`
**Description**: Delete a document and its associated Qdrant chunks, updating `state.json`.

**Request**:
- **Parameters** (path/query):
  - `file_id` (str): The document ID to delete.
  - `user_id` (str): Unique identifier for the user.
- **Example**:
  ```bash
  curl -X DELETE "http://localhost:8000/documents/uuid-string?user_id=test_user"
  ```

**Response**:
- **Status**: 200 OK
- **Body**:
  ```json
  {
    "status": "success",
    "file_id": "uuid-string"
  }
  ```
- **Errors**: 500 (Qdrant or state file failure).

## Streamlit Frontend
The Streamlit interface (`http://localhost:8501`) provides:
- **Document Management**: Upload files, view document list, select documents for context, preview files, and delete documents.
- **Chat Interface**: Create chat sessions, send queries, and view responses with source citations.
- **Debug Interface**: Inspect file metadata and Qdrant chunks for a specific document (`?page=debug&file_id=uuid-string`).
- **State Persistence**: Syncs with FastAPI via `app/data/state.json` for `file_metadata` and `chat_sessions`.

**Usage**:
1. Open `http://localhost:8501`.
2. Enter a `user_id` (e.g., `test_user`).
3. Upload a file (e.g., `notes.txt`, `document.pdf`).
4. Select documents via checkboxes for chat context.
5. Create a chat session and send queries.
6. Use the debug page to inspect Qdrant chunks and metadata.
7. Test cross-service consistency (e.g., upload via FastAPI, view in Streamlit).

## Troubleshooting
- **Containers Not Running**:
  ```bash
  docker ps
  docker logs pdfllm-rag-service-1
  docker logs pdfllm-streamlit-service-1
  docker logs pdfllm-qdrant-1
  ```
  - Ensure `app/main.py`, `app/streamlit_app.py`, and `requirements.txt` exist.
  - Verify `OPENAI_API_KEY` in `.env`.

- **State Sharing Issues**:
  - Check `app/data/state.json`:
    ```bash
    cat app/data/state.json
    chmod -R 777 app/data
    ```
  - Ensure both services write to `/app/data/state.json`.

- **FastAPI Errors**:
  - If endpoints fail, try `gpt-3.5-turbo` in `app/main.py`:
    ```python
    model="gpt-3.5-turbo"
    ```
  - Upgrade to `openai==1.40.0` for async support:
    ```bash
    echo "openai==1.40.0" >> requirements.txt
    docker-compose up --build
    ```

- **Streamlit Issues**:
  - If documents don’t appear, check:
    ```python
    print(st.session_state.file_metadata)
    ```
  - If checkboxes fail, verify:
    ```python
    print(st.session_state.selected_docs)
    ```

- **PDF Headings**:
  - If Qdrant chunks show generic `# PDF Content` in `parent_section`, check `app/converters/pdf_converter.py`. Update it to extract meaningful headings (e.g., from PDF metadata or content structure).

## Known Issues
- **Generic PDF Headings**: Chunks may have `# PDF Content` in `parent_section`. To fix, modify `app/converters/pdf_converter.py` to extract specific headings. Share the file for assistance.
- **OpenAI Sync Calls**: `openai==0.27.8` uses synchronous calls, which may slow performance. Upgrade to `openai==1.40.0` for async support if needed.

## Contributing
- Report issues or suggest features via the repository’s issue tracker.
- To fix PDF headings, share `app/converters/pdf_converter.py` for targeted improvements.

## License
MIT License (or specify your preferred license).