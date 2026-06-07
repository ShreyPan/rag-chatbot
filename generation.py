import ollama


def generate_answer(question, relevant_chunks, chat_history):
    context = "\n\n".join(relevant_chunks)

    system_prompt = f"""You are a helpful assistant. You will be given some context retrieved from documents and a question.

    If the context contains relevant information to answer the question, use it to answer.
    If the context is not relevant to the question, answer from your general knowledge.
    Never mention that you are using context or general knowledge in your answer.

    Context:
    {context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history
    messages.append({"role": "user", "content": question})

    response = ollama.chat(
        model="llama3.2",
        messages=messages
    )

    return response["message"]["content"]
