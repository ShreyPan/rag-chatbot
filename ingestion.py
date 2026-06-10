import os

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdfs(folder_path):
    # returns a dict: {filename: full_text} instead of a flat list
    files = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
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

    return files


def split_into_chunks(files):
    # returns a dict: {filename: [chunks]}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
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
    client = chromadb.PersistentClient(path="chroma_store")

    embedding_fn = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434/api/embeddings"
    )

    collection = client.get_or_create_collection(
        name="documents",
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
