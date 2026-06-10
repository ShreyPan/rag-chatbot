# RAG Agent

A fully local, privacy-focused agentic AI system built with Python. Uses a ReAct loop to decide between searching loaded PDF documents or the web to answer questions, with conversation memory across the session.

## What it does

- Loads multiple PDFs from a `docs/` folder
- Chunks and embeds documents using a local embedding model
- Stores vectors persistently in ChromaDB, skipping already ingested files
- Agent autonomously decides whether to search documents or the web based on the question
- Maintains conversation history so follow-up questions work naturally
- Runs fully locally, no API keys required

## How it works
User question
↓
Agent (ReAct loop)
↓ thinks: which tool do I need?
↓
Tool 1: search_documents → searches ChromaDB for relevant chunks
Tool 2: search_web       → searches DuckDuckGo for current info
↓
Agent observes result, decides if answer is sufficient
↓
Final answer

## Tech Stack

- **LLM**: llama3.2 via Ollama (runs locally)
- **Embeddings**: nomic-embed-text via Ollama
- **Vector DB**: ChromaDB (persistent local storage)
- **Web Search**: DuckDuckGo via ddgs
- **Chunking**: LangChain RecursiveCharacterTextSplitter
- **PDF parsing**: pypdf

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
├── ingestion.py     # PDF loading, chunking, per-file ChromaDB storage with metadata
├── retrieval.py     # semantic similarity search
├── generation.py    # ReAct agent loop with tool calling
├── tools.py         # document search and web search tools
├── requirements.txt # dependencies

## Design Decisions

- Built the agent loop manually using raw Ollama and ChromaDB calls instead of LangChain abstractions, to understand what frameworks like LangChain hide under the hood
- Per-file ingestion tracking means adding new PDFs to the docs/ folder automatically ingests only the new files on next run
- Page text is joined before chunking to avoid context being split at page boundaries
- Chunk metadata stores source filename and chunk index for traceability