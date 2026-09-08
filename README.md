# Grounded — Local RAG Knowledge Base

A local document question-answering application built with
Python, FastAPI, ChromaDB, and Ollama.

## Features

- Upload PDF, TXT, and Markdown documents
- Generate embeddings locally with Ollama
- Persist searchable document chunks in ChromaDB
- Ask questions and inspect retrieved source excerpts
- List and delete indexed documents
- Responsive web interface and interactive API documentation
- Unit tests and GitHub Actions

## Architecture

Document -> Extract text -> Chunk -> Embed -> ChromaDB

Question -> Embed -> Retrieve excerpts -> Local LLM -> Answer with sources

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

Upload sample-policy.txt and ask:
How many unused leave days can I carry over?

If Ollama is not running, start ollama serve in another terminal.

## Tests

    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
    .\.venv\Scripts\python.exe -m pytest -q

Tests cover text processing and input validation, not real model accuracy.

## Configuration

The application reads these environment variables directly:

- OLLAMA_URL: http://localhost:11434
- CHAT_MODEL: llama3.2:3b
- EMBED_MODEL: nomic-embed-text
- DATA_DIR: data folder beside app.py

It does not automatically load .env files.
Changing the embedding model requires re-uploading documents into the new index.

## Limitations and security

This is a local, single-user starter, not a production internet service.

- No authentication, document-level permissions, or rate limiting.
- Maximum upload: 10 MB; PDF: 200 pages; document: 500 chunks.
- Scanned PDFs require OCR. Encrypted PDFs are unsupported.
- Upload size is checked after multipart parsing.
- Chunking is character-based; processing is synchronous.
- Questions are independent; conversation history is not sent to the model.
- No reranking or calibrated retrieval threshold.
- AI answers and citations may be incorrect; verify the excerpts.
- Prompt instructions do not eliminate document-based prompt injection.
- Extracted text and embeddings are stored without application-level encryption.
- A remote OLLAMA_URL sends document text to that remote server.
- Deletion is not guaranteed secure erasure.
- The health endpoint does not check Ollama readiness.

Do not expose this application directly to the internet.
Do not commit .env, .venv, data, or private documents.

## Roadmap

- OCR support
- Token-aware chunking
- Hybrid retrieval and reranking
- Streaming answers
- Background indexing
- Authentication and document permissions