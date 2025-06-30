# pdfLLM

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice designed for processing, storing, and querying documents such as PDFs, text files, Word documents, spreadsheets, and images. It leverages a FastAPI backend for programmatic access, a Streamlit frontend for interactive use, and a Qdrant vector database for efficient retrieval. Documents are converted to markdown, chunked, and embedded using OpenAI’s `text-embedding-3-small` model, with entities and relationships indexed in a knowledge grap...

## Overview

`pdfLLM` is a hybrid RAG application that combines **semantic search** (vector-based similarity) with **graph-based search** (entity-relationship traversal) to provide accurate and contextually relevant answers. It processes uploaded documents, extracts text, identifies entities and relationships, and stores them in Qdrant and a `networkx`-based knowledge graph. Users can interact via a web UI or API, uploading files, querying document content, and debugging stored data. The system is designed for s...

## Deployment

```bash
git clone https://github.com/ikantkode/pdfLLM.git
cd pdfLLM
mv env_example .env
# Put your OpenAI key in .env
docker compose up -d --build
```

- Visit **http://localhost:8501** for the Streamlit web interface.
- View `api_docs.md` for more information on using the FastAPI endpoints.

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
- **Semantic Search**:
  - **Mechanism**: Uses OpenAI’s `text-embedding-3-small` to generate 1536-dimensional embeddings for document chunks and queries. Qdrant performs cosine similarity searches to retrieve the top `limit` (default 5) chunks.
  - **Implementation**: The `qdrant_handler.py` `search_entities` method filters chunks by `user_id`, `file_id` (optional), and entities, combining vector search results with `rank_results` in `main.py`.
  - **Example**: Querying “What is the requisition amount for PS 54X?” retrieves chunks containing “CURRENT PAYMENT DUE: $29,825.80” based on semantic similarity.
- **Graph-Based Search**:
  - **Mechanism**: Extracts entities (e.g., “Varsity Plumbing and Heating, Inc.”) and relationships (e.g., “appears_in”) using Spacy (`en_core_web_sm`) in `text_processor.py`. Stores them in a `networkx` graph (`knowledge_graph.json`) and Qdrant payloads.
  - **Implementation**: The `/search` endpoint in `main.py` uses `search_documents` to combine vector search with entity-based filtering via `qdrant_handler.search_entities`. The knowledge graph enhances queries targeting entities or relationships.
  - **Example**: Querying “Who is the subcontractor for PS 54X?” retrieves chunks linked to “Varsity Plumbing and Heating, Inc.” via the “appears_in” relationship.
- **Hybrid Approach**: Combines semantic and graph search results in `rank_results`, prioritizing chunks with high similarity and relevant entities/relationships.

## License
MIT License

## API Documentation
See `api_docs.md` for FastAPI endpoint details.