from ingestion import load_pdfs, split_into_chunks, store_in_chromadb
from generation import run_agent
from config import DOCS_FOLDER

# 1. Setup

files = load_pdfs(DOCS_FOLDER)
chunks_by_file = split_into_chunks(files)
collection = store_in_chromadb(chunks_by_file)

# 2. Chat loop

chat_history = []

print("\nRAG Chatbot ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    answer = run_agent(question, collection, chat_history)

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"Assistant: {answer}\n")
