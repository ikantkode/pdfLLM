# pdfLLM - Document Processing and Retrieval-Augmented Generation

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice designed for processing, embedding, and querying documents. It integrates document parsing, semantic search, knowledge graph construction, and LLM-based generation to provide advanced document intelligence. Built with scalability in mind, it uses PostgreSQL for session management, Qdrant for vector storage, Dgraph for graph-based indexing, and Celery for asynchronous task processing.

## Change Log

- **07/31/2025**: 
  - Implemented smaller chunk sizes (500 tokens) for improved embedding accuracy and retrieval performance.
  - Integrated Celery for asynchronous OCR processing and file uploads, enabling faster handling of multiple files (tested with up to 5 files simultaneously).
  - Replaced `state.json` with PostgreSQL for robust session and metadata management.
  - Introduced "Categories" feature, allowing users to create custom categories with tailored prompts for document organization.
  - Added "Master Chat" and "Category Chat" features for querying across all documents or specific categories, with orchestration chain in progress.
  - PostgreSQL runs on a non-standard port to avoid conflicts with existing installations.
- **07/27/2025**: Initial support for asynchronous multi-file uploads and category-based prompts.

## Context Size of Models

The application supports multiple embedding and chat models, with a focus on cost-effective and high-performance options. For consistency, embeddings are truncated to 1,024 dimensions, and documents are chunked into 500-token segments for processing.

| Provider    | Model                       | Dimensions | Max Tokens | Price ($ / 1M tokens) |
|-------------|-----------------------------|------------|------------|-----------------------|
| **OpenAI**  | text-embedding-3-small      | 1,536      | ~8,191     | $0.020                |
|             | text-embedding-3-large      | 3,072      | ~8,191     | $0.130                |
| **Google**  | gemini-embedding-001        | 3,072      | 2,048      | *Not disclosed*       |
|             | text-embedding-preview-0409 | 768        | 2,048      | $0.025                |
| **Mistral** | mistral-embed               | 1,024      | 32,768     | $0.010                |

**Recommendation**: OpenAI's `text-embedding-3-small` and `gpt-4o-mini` are highly cost-effective and reliable for most use cases. For large documents, chunks are processed within an 8,000-token limit to ensure compatibility with Qdrant and Dgraph.
**As of 07/31/2025**: I want to try in Qwen3-Embedding-0.6B and Qwen3-30B-A3B (non-thinking/instruct) for retrieval. I am legitimately flabbergasted at the performance. The Qwen team done out did it all. VLLM would have openAI compatible endpoints and essentially, this would be a plug and play. If I do it, I will share docker stuff for it.

### Example Use Case
Organize a business's document corpus (e.g., inventory receipts, payroll, utility bills) into categories. Use category-specific prompts like "Summarize inventory spending" to generate modular summaries stored in PostgreSQL. Combine these summaries with a master prompt (e.g., "How much was spent on the store?") using `gpt-4o-mini` (128,000-token context window) for a coherent, hybrid response.

## Overview

pdfLLM combines:
- 📚 **Semantic Search**: Embedding-based retrieval using OpenAI models.
- 🧠 **Graph-Based Search**: Entity and relationship indexing via Dgraph.
- 💬 **LLM-Powered Responses**: Accurate, cited answers using OpenAI chat models.
- 📊 **Hybrid Retrieval**: Combines vector similarity and entity relationships for precise results.

Documents are parsed, cleaned (OCR-aware), chunked into 500-token segments, embedded, and indexed in Qdrant (vectors) and Dgraph (entities/relationships).

## Features

- 🗃 **Supported Formats**: `.pdf`, `.txt`, `.doc(x)`, `.xls(x)`, `.csv`, `.jpg`, `.png`, `.heic`, `.webp`, `.md`, `.rtf`, `.odt`, `.ods`
- 🔄 **Conversion**: Converts documents to markdown using specialized parsers.
- ✂️ **Chunking & Embedding**: Tokenizes and chunks markdown into 500-token segments; embeddings generated via OpenAI.
- 🧾 **Metadata Storage**: Stores file metadata and base64 content in PostgreSQL for previews.
- 🔍 **Search**: `/search` endpoint supports hybrid semantic and graph-based queries.
- 💬 **Chat**: `/chat` endpoint provides answers with cited sources, supporting Master Chat (all documents) and Category Chat (category-specific).
- 🧠 **Knowledge Graph**: `/knowledge_graph` exposes nodes and edges for advanced querying.
- 🔒 **Security**: All endpoints require `X-API-Key` authentication.
- 🚀 **Asynchronous Processing**: Celery handles OCR and file uploads for improved performance.
- 👁 **Preview**: Preview uploaded files via `/preview/{file_id}`.
- 🗂 **Categories**: Organize documents into user-defined categories with custom prompts.

## Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/ikantkode/pdfLLM.git
   cd pdfLLM
   ```
2. Configure environment variables:
   ```bash
   cp env_example .env
   ```
   Update `.env` with your OpenAI API key, PostgreSQL settings, and other configurations.
3. Launch the application:
   ```bash
   docker compose up -d --build
   ```

## Example Use Cases

- Extract payroll details from scanned PDFs and summarize hours worked (e.g., "How many hours did the carpenter work in April?").
- Summarize project submissions or funding reports across multiple documents.
- Organize business documents into categories (e.g., inventory, payroll) and query specific categories or the entire knowledge base.

## Current Issues

- Large PDFs (>40 pages) may face context window limitations, impacting processing accuracy.
- Quantized models via Ollama (e.g., on a 3060 with 12GB VRAM) may produce suboptimal results, such as font rendering issues.
- Multi-file processing is tested up to 5 files; further testing for larger batches is ongoing.
- Orchestration chain for Master Chat is still under development, which may affect response coherence for complex queries.
- Evaluation framework for RAG performance is not yet implemented.
- Single system prompt limitation requires manual prompt adjustments for specific use cases.

## Roadmap

- ✅ OCR-aware chunk cleaning
- ✅ Graph-enhanced search results
- ✅ Replaced `state.json` with PostgreSQL
- ✅ Celery for asynchronous processing
- ✅ Ollama / Local LLM support
- 🔜 JWT authentication (not planned for base pdfLLM)
- 🔜 Dynamic model selection (OpenAI, DeepSeek, Grok) (not planned for base pdfLLM)
- 🔜 Enhanced orchestration for Master Chat

## License

MIT License

## API Documentation

See [`api_docs.md`](./api_docs.md) for detailed endpoint usage.