import os

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_pdfs(folder_path):
    all_text = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            reader = PdfReader(path)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)

            print(f"Loaded: {filename}")
    return all_text


def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.create_documents(pages)
    return chunks


def store_in_chromadb(chunks):
    client = chromadb.PersistentClient(path="chroma_store")

    embedding_fn = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434/api/embeddings"
    )

    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )

    if collection.count() > 0:
        print(
            f"Collection already has {collection.count()} chunks, skipping embedding")
        return collection

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            ids=[f"chunk_{i}"]
        )

    print(f"Stored {len(chunks)} chunks in ChromaDB")
    return collection
