from ingestion import load_pdfs, split_into_chunks, store_in_chromadb
from retrieval import retrieve
from generation import generate_answer

# 1. Setup

docs = load_pdfs("docs")
chunks = split_into_chunks(docs)
collection = store_in_chromadb(chunks)

# 2. Chat loop

chat_history = []

print("\nRAG Chatbot ready. Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    relevant_chunks = retrieve(question, collection)
    answer = generate_answer(question, relevant_chunks, chat_history)

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    print(f"Assistant: {answer}\n")
