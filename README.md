# pdfLLM

`pdfLLM` is a Retrieval-Augmented Generation (RAG) microservice that processes documents to allow querying, summarization, and chat-like interaction. It integrates document parsing, knowledge graph construction, semantic search, and LLM-based generation to offer advanced document intelligence.

## Change Log

- As of today, 07/27/2025, the app is now able to asynchronously upload/process multiple files. (Tested up to 5 more to be tested tomorrow).
- Introducing "Categories" - you are now able to create categories for your application, and are able to create prompts custom to that category. 
- Introducing "Master Chat" and "Category Chat" - you are able to isolate chats to specific categories, and or initiate chats with the entirety of your "Knowledge Base"; although orchestration chain is still to be implemented, it "works".
- Postgres is now used instead of state.json to manage and cache your sessions. Its all good baby baby.
- Postgres docker conatiner runs on a different than usual port so if you have your own postgre installation, it won't mess with it.
- Celery for the win. Processing is blazing fast now. Kinda surprised tbh.
- OCR improvements are coming. I can honestly work on this for the rest of my life and there would still be improvements. So help. pls. {insert kevin hart stage meme here}

## Context Size of Models (Total, Input and Output)

Using quantized models via Ollama will *severely* output the level of results you want. Context windows are therefore undetereminable for everyone's setup, but I can safely recommend OpenAI's gpt-4o-mini and text-embedding-3-small models as so cost effective that it is almost a no-brainer to use. 

| Provider    | Model                       | Dimensions | Max Tokens | Price (\$ / 1M tokens)           |
| ----------- | --------------------------- | ---------- | ---------- | -------------------------------- |
| **OpenAI**  | text‑embedding‑3‑small      | 1,536      | \~8,191    | **\$0.020**  |
|             | text‑embedding‑3‑large      | 3,072      | \~8,191    | **\$0.130**                      |
| **Google**  | gemini‑embedding‑001        | 3,072      | 2,048      | *Pricing not disclosed*          |
|             | text‑embedding‑preview‑0409 | 768        | 2,048      | **\$0.025**                      |
| **Mistral** | mistral‑embed               | 1,024      | 32,768     | **\$0.010**                      |

We are truncating dimensions to 1,024 so that we have consistency across the board and we are also limiting the max tokens per embedding request to 8,000 - this way if a document is large and has biggeer context window, it is sent in chunks to be processed into vector embeddings which is then saved into qdrant and dgraph accordingly.

I think the best way to visualize tokens is to go to any LLM (ChatGPT/DeepSeek/Meta.ai) and request each chat instance to give a story in 1024 tokens, 2048, 4096, 8,000 and you will see just the amount of words that come out. If you have a RAG app deployed, you are easily able to orchestrate between multiple modules of your SaaS App, have separate summariziations of different documents, and then connect them all together for a summary.

For example:

If you have a large corpus, you would ideally want to organize it. Let's imagine we are *any* business:

    - Purchase of Inventory
    - Receipts of sales (or export if using software - some business explicitly accept cash)
    - Payroll of employees
    - Large purchases
    - Utility Bills
    - etc.

You would ideally want to have seperate modules for each type of data, then run separate summary for inventory. So that prompt would look a little something like "Thoroughly go through each inventory receipt and summarize how much was spent in inventory." - now of course in a real world application this would be a deeper prompt, but this would yield a summary. You would repeat this step for each module and that would be fed into the redis cache or something similar. From that redis cache, we can hypothetically retrieve the context necessary - and since the output would be limited to either 4,000 or 8,000 tokens, we can safely feed it into something like gpt-4o-mini that has a limit of 128,000 context window; which would then give us a nice and coherent answer to the user's query of something like, "Can you tell me how much money I spent on the store?" - the result would be a combination of hybrid response based off the user's context.

## The "Safety"

If you think OpenAI is evil, then just know, your iPhone/Android is listening to your conversations (hence the oddly accurate ads about the conversation you just had), and how much stuff the companies silently have on you is kind of insane. Make your life better/easier. You're on this earth for a limited time.

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
- Current state of the app is technically limited to "single system prompt" for retrieval. This means you must adjust the prompt to cater to your needs, otherwise the reponse is conjoined text/funny text. The retrieval is still accurate.

## Roadmap

- ✅ OCR-aware chunk cleaning
- ✅ Graph-enhanced search results
- ✅ Replace networkx with dgraph
- ✅ Ollama / Local LLM support
- 🔜 JWT authentication (no plans to implement it into base-pdfLLM)
- 🔜 Dynamic embedding/chat model selection (OpenAI, DeepSeek, Grok) (no plans to implement in base-pdfLLM)

## License

MIT License

## API Documentation

See [`api_docs.md`](./api_docs.md) for full endpoint usage.
