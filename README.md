# pdfLLM

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice that processes documents to allow querying, summarization, and chat-like interaction. It integrates document parsing, knowledge graph construction, semantic search, and LLM-based generation to offer advanced document intelligence.

## Overview

The system combines:
- 📚 **Semantic Search**: Embedding-based retrieval using OpenAI.
- 🧠 **Graph-Based Search**: Entity and relationship indexing via dgraph.
- 💬 **LLM-Powered Answers**: Uses OpenAI chat models to generate accurate, cited responses.
- 📊 **Hybrid Retrieval**: Combines vector similarity and entity relationships for precision.

Documents are parsed, cleaned (OCR-aware), chunked, embedded, and indexed both in Qdrant and a dgraph.

## Features

- 🗃 **Supported Formats**: `.pdf`, `.txt`, `.doc(x)`, `.xls(x)`, `.csv`, `.jpg`, `.png`, `.heic`, `.webp`, `.md`, `.rtf`, `.odt`, `.ods`
- 🔄 **Conversion**: Converts to markdown using specialized parsers.
- ✂️ **Chunking & Embedding**: Tokenizes and chunks cleaned markdown; embeddings generated via OpenAI.
- 🧾 **Metadata Storage**: File metadata + base64 content saved for previews.
- 🔍 **Search**: Hybrid search via `/search` endpoint (semantic + graph-based).
- 💬 **Chat**: `/chat` endpoint answers queries with sources cited by section.
- 🧠 **Knowledge Graph**: `/knowledge_graph` exposes nodes and edges.
- 🔒 **Security**: All endpoints require `X-API-Key`.
- 📁 **Persistent State**: `state.json` and `knowledge_graph.json` are stored for resilience.
- 👁 **Preview**: Preview uploaded files directly via `/preview/{file_id}`.

## Deployment

```bash
git clone https://github.com/ikantkode/pdfLLM.git
cd pdfLLM
mv env_example .env  # add your OpenAI and Qdrant configs
docker compose up -d --build
```

## Example Use Cases

- Extract payroll details from messy scanned PDFs.
- Summarize project submissions and funding reports.
- Ask structured questions like "How many hours did the carpenter work in April?"

## Roadmap

- ✅ OCR-aware chunk cleaning
- ✅ Graph-enhanced search results
- ✅ Replace networkx with dgraph
- 🔜 Ollama / Local LLM support
- 🔜 JWT authentication
- 🔜 Dynamic embedding model selection (OpenAI, DeepSeek, Grok)

## License

MIT License

## API Documentation

See [`api_docs.md`](./api_docs.md) for full endpoint usage.
