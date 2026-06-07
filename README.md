# RAG Chatbot

A fully local, privacy-focused RAG (Retrieval Augmented Generation) chatbot built with Python. Query multiple PDF documents using natural language, with conversation memory across questions.

## What it does

- Loads multiple PDFs from a `docs/` folder
- Chunks and embeds documents using a local embedding model
- Stores vectors persistently in ChromaDB
- Retrieves relevant chunks based on semantic similarity
- Generates answers using a local LLM (no API key required)
- Maintains conversation history so follow-up questions work naturally

## Tech Stack

- **LLM**: llama3.2 via Ollama (runs locally)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector DB**: ChromaDB
- **Chunking**: LangChain RecursiveCharacterTextSplitter
- **PDF parsing**: pypdf

## Architecture
docs/ (PDFs)
↓ ingestion.py
Chunked + Embedded → ChromaDB
↓ retrieval.py
Semantic search → relevant chunks
↓ generation.py
llama3.2 → answer
↓ main.py
Chat loop with memory

## Setup

1. Install Ollama from https://ollama.com and pull the required models:
ollama pull nomic-embed-text
ollama pull llama3.2

2. Clone the repo and install dependencies:
pip install -r requirements.txt

3. Add your PDF files to a `docs/` folder

4. Run:
python main.py

## Project Structure
├── main.py          # entry point and chat loop
├── ingestion.py     # PDF loading, chunking, ChromaDB storage
├── retrieval.py     # semantic search
├── generation.py    # LLM answer generation
├── requirements.txt # dependencies
