# Grounded — Local RAG Knowledge Base

A document-grounded question-answering application built with Python, FastAPI, ChromaDB, and Ollama.

## 🚀 Live UI

The web application is deployed on Render:

**https://grounded-rag-oww5.onrender.com**

> The public Render instance provides the UI/API, but the **no-API-key AI mode runs Ollama locally**. This avoids OpenAI/Hugging Face billing and API keys.

## Features

- Upload PDF, TXT, and Markdown documents
- Generate embeddings locally with Ollama
- Persist searchable document chunks in ChromaDB
- Ask questions and inspect retrieved source excerpts
- Source-grounded answers with `[S1]`, `[S2]` citations
- List and delete indexed documents
- Responsive web interface
- Interactive API documentation
- Unit tests and GitHub Actions
- Optional OpenAI-compatible cloud provider support when credentials are available

## No-API-Key Mode 🔐

Grounded RAG is designed to work without an OpenAI or Hugging Face API key by using **Ollama locally**. Ollama's local API does not require authentication and is served at `http://localhost:11434` by default. citehttps://docs.ollama.com/api/authentication

Architecture:

    Browser -> FastAPI -> ChromaDB
                       -> Ollama (local, no API key)

## Architecture

    Document -> Extract text -> Chunk -> Local Ollama embeddings -> ChromaDB

    Question -> Local Ollama embedding -> Retrieve excerpts -> Local Ollama LLM -> Answer with sources

## Requirements

- Python 3.11 or 3.12
- Ollama installed from https://ollama.com
- Enough RAM and disk space for the selected models

## Windows quick start

Run these commands from the project folder:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

With Ollama running, download the models:

    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

Start the application:

    .\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000

Open http://127.0.0.1:8000 in your browser.

API documentation: http://127.0.0.1:8000/docs

Upload `sample-policy.txt` and ask:

    How many unused leave days can I carry over?

If Ollama is not running, start it with `ollama serve` in another terminal. Ollama's local API is available at `http://localhost:11434`. citehttps://docs.ollama.com/api/introduction

## Tests

    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    .\.venv\Scripts\python.exe -m pytest -q

Tests cover text processing and input validation, not real model accuracy.

## Configuration

The application reads these environment variables directly:

- `AI_PROVIDER`: defaults to `ollama`
- `OLLAMA_URL`: defaults to `http://localhost:11434`
- `CHAT_MODEL`: defaults to `llama3.2:3b`
- `EMBED_MODEL`: defaults to `nomic-embed-text`
- `DATA_DIR`: defaults to the `data` folder beside `app.py`
- `AI_BASE_URL`: optional OpenAI-compatible provider endpoint
- `AI_API_KEY`: optional cloud-provider credential
- `APP_API_KEY`: optional application API protection

It does not automatically load `.env` files.
Changing the embedding model requires re-uploading documents into the new index.

## Deployment note

The Render deployment is intentionally kept free of provider credentials. The application remains deployed and accessible as a web UI, while full no-key AI inference is performed by Ollama on a machine where the model is installed. Ollama's local API requires no authentication. citehttps://docs.ollama.com/api/authentication

For a fully cloud-hosted AI experience accessible to everyone without the user's computer running Ollama, a hosted inference provider or dedicated model server is required.

## Limitations and security

This is primarily a local, single-user RAG application, not a production internet service.

- No authentication, document-level permissions, or rate limiting unless `APP_API_KEY` is configured.
- Maximum upload: 10 MB; PDF: 200 pages; document: 500 chunks.
- Scanned PDFs require OCR. Encrypted PDFs are unsupported.
- Upload size is checked after multipart parsing.
- Chunking is character-based; processing is synchronous.
- Questions are independent; conversation history is not sent to the model.
- No reranking or calibrated retrieval threshold.
- AI answers and citations may be incorrect; verify the excerpts.
- Prompt instructions do not eliminate document-based prompt injection.
- Extracted text and embeddings are stored without application-level encryption.
- A remote `OLLAMA_URL` sends document text to that remote server.
- Deletion is not guaranteed secure erasure.
- The health endpoint does not check Ollama readiness.

Do not expose an Ollama server directly to the public internet without appropriate network controls.
Do not commit `.env`, `.venv`, `data`, API keys, or private documents.

## Roadmap

- OCR support
- Token-aware chunking
- Hybrid retrieval and reranking
- Streaming answers
- Background indexing
- Authentication and document permissions
- Optional browser-local inference for a zero-key hosted demo
