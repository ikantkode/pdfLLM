# pdfLLM

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice that processes documents to allow querying, summarization, and chat-like interaction. It integrates document parsing, knowledge graph construction, semantic search, and LLM-based generation to offer advanced document intelligence.

## Overview

The system combines:
- 📚 **Semantic Search**: Embedding-based retrieval using OpenAI.
- 🧠 **Graph-Based Search**: Entity and relationship indexing via dgraph.
- 💬 **LLM-Powered Answers**: Uses OpenAI chat models to generate accurate, cited responses.
- 📊 **Hybrid Retrieval**: Combines vector similarity and entity relationships for precision.

Documents are parsed, cleaned (OCR-aware), chunked, embedded, and indexed both in Qdrant and dgraph.

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
mv env_example .env
docker compose up -d --build
```

## Example Use Cases

- Extract payroll details from messy scanned PDFs.
- Summarize project submissions and funding reports.
- Ask structured questions like "How many hours did the carpenter work in April?"

## Current Issues:

- Larger PDFs might be troublesome to process due to limited context window.
- My Ollama on a 3060 (12GB VRAM) can only run Q4 embedding and chat models, thus, the results are not as clear (fonts may come funny) so don't be surprised.
- Tests have only been conducted with single PDFs (upto 40 pages) - context limits are an issue. 
- I dont know how to run evaluations on this rag app yet.
- See below for an issue due to my exhaustion.
    - Currently, you must update config.py in addition to the .env to propagate your settings (for example: if you change your model). 

    ### OpenAI settings
    openai_enabled: bool = Field(False, env="OPENAI_ENABLED")
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field("gpt-4o-mini", env="OPENAI_CHAT_MODEL")

    ### Ollama settings
    ollama_enabled: bool = Field(False, env="OLLAMA_ENABLED")
    ollama_host: str = Field("localhost", env="OLLAMA_HOST")
    ollama_port: int = Field(11434, env="OLLAMA_PORT")
    ollama_embedding_model: str = Field("bge-m3:latest", env="OLLAMA_EMBEDDING_MODEL")
    ollama_chat_model: str = Field("llama3.1:8b", env="OLLAMA_CHAT_MODEL")

## Roadmap

- ✅ OCR-aware chunk cleaning
- ✅ Graph-enhanced search results
- ✅ Replace networkx with dgraph
- ✅ Ollama / Local LLM support
- 🔜 JWT authentication
- 🔜 Dynamic embedding/chat model selection (OpenAI, DeepSeek, Grok)

## License

MIT License

## API Documentation

See [`api_docs.md`](./api_docs.md) for full endpoint usage.
