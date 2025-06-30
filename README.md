# pdfLLM

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice designed for processing, storing, and querying documents such as PDFs, text files, Word documents, spreadsheets, and images. It leverages a FastAPI backend for programmatic access, a Streamlit frontend for interactive use, and a Qdrant vector database for efficient retrieval. Documents are converted to markdown, chunked, and embedded using OpenAI’s `text-embedding-3-small` model, with entities and relationships indexed in a knowledge graph. The system supports semantic search for context-based retrieval and graph-based search for entity-relationship queries, enabling precise and context-aware responses using OpenAI’s `gpt-4o` model.

## Overview

`pdfLLM` is a hybrid RAG application that combines **semantic search** (vector-based similarity) with **graph-based search** (entity-relationship traversal) to provide accurate and contextually relevant answers. It processes uploaded documents, extracts text, identifies entities and relationships, and stores them in Qdrant and a `networkx`-based knowledge graph. Users can interact via a web UI or API, uploading files, querying document content, and debugging stored data. The system is designed for scalability, persistence, and cross-service state sharing, making it suitable for document-heavy workflows like project management, legal analysis, or research.

### Type of RAG Application
- **Retrieval-Augmented Generation**: Combines retrieval (fetching relevant document chunks) with generation (producing natural language responses using OpenAI’s `gpt-4o`).
- **Hybrid Retrieval**: Integrates semantic search (cosine similarity on embeddings) with graph-based search (entity-relationship queries), enhancing precision for structured queries.
- **Use Case**: Ideal for querying specific details (e.g., “What is the requisition amount for Project A?”) or summarizing documents (e.g., “What is this about?”) with source citations.

## Features
- **Document Processing**: Converts files (`.pdf`, `.txt`, `.doc`, `.docx`, `.odt`, `.xls`, `.xlsx`, `.csv`, `.ods`, `.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`, `.md`, `.rtf`) to markdown using specialized converters.
- **Vector Storage**: Stores document chunks in Qdrant with OpenAI embeddings (`text-embedding-3-small`) for semantic search.
- **Semantic Search**: Retrieves chunks based on cosine similarity between query and chunk embeddings, ensuring contextually relevant results.
- **Graph-Based Search**: Indexes entities (e.g., “Example Construction Inc.”) and relationships (e.g., “appears_in”) in a `networkx` graph, enabling structured queries.
- **Knowledge Graph**: Persists entities and relationships in `knowledge_graph.json`, supporting queries like “Who is the subcontractor for Project A?”.
- **Querying**: Generates responses using OpenAI’s `gpt-4o` model, citing sources (filename, section) for transparency.
- **FastAPI Backend**: Provides programmatic access for file processing, searching, chatting, and document management.
- **Streamlit Frontend**: Offers a UI for uploading files, managing documents, chatting, and debugging Qdrant chunks and metadata.
- **State Persistence**: Shares `file_metadata` and `chat_sessions` between services via `state.json` and `streamlit_state.json`.
- **Debug Interface**: Inspects file metadata and Qdrant chunks via `?page=debug&file_id=<uuid>`.

## Semantic and Graph Search
... (truncated for brevity, see original content)

## License
MIT License

## API Documentation
See `api_docs.md` for FastAPI endpoint details.