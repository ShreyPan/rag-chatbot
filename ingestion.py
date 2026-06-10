import os

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_URL


def load_pdfs(folder_path):

    files = {}

    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist. Creating it.")
        os.makedirs(folder_path)
        return files

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"Warning: No PDF files found in '{folder_path}'.")
        return files

    for filename in pdf_files:
        try:
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)

            # join all pages into one string, fixes page boundary problem
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)

            files[filename] = "\n".join(pages)
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}, skipping.")

    return files


def split_into_chunks(files):
    # returns a dict: {filename: [chunks]}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks_by_file = {}
    for filename, text in files.items():
        chunks_by_file[filename] = splitter.create_documents([text])

    return chunks_by_file


def get_ingested_files(collection):
    existing = collection.get()["ids"]
    if not existing:
        return set()
    return set(id.split("_chunk_")[0] for id in existing)


def store_in_chromadb(chunks_by_file):
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        embedding_fn = OllamaEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            url=OLLAMA_URL
        )

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )

        ingested = get_ingested_files(collection)

        for filename, chunks in chunks_by_file.items():
            if filename in ingested:
                print(f"Already ingested: {filename}, skipping")
                continue

            for i, chunk in enumerate(chunks):
                collection.add(
                    documents=[chunk.page_content],
                    ids=[f"{filename}_chunk_{i}"],
                    metadatas=[{"source": filename, "chunk": i}]
                )
            print(f"Ingested: {filename} ({len(chunks)} chunks)")

        return collection

    except Exception as e:
        print(f"Error connecting to ChromaDB or Ollama: {e}")
        print("Make sure Ollama is running: run 'ollama serve' in a separate terminal.")
        exit(1)
