import ollama  # type: ignore
import json


def run_agent(question, collection, chat_history):
    from tools import search_documents, search_web

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "ALWAYS use this tool first for any question about a person, their projects, experience, skills, education, or work history. Searches the loaded PDF documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Use this tool ONLY for general knowledge questions, definitions of concepts, or current events that would NOT be found in a personal document like a resume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Build messages with history
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You have access to two tools: search_documents and search_web. ALWAYS use search_documents first for ANY question that could be about a person, their work, or a named project. Only use search_web for pure concept definitions or current events. When answering from documents, always mention the source file at the end of your answer."}
    ]
    messages += chat_history
    messages.append({"role": "user", "content": question})

    # ReAct loop
    max_iterations = 5
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        try:
            response = ollama.chat(
                model="llama3.2",
                messages=messages,
                tools=tools
            )
        except Exception as e:
            return f"Error communicating with Ollama: {e}. Make sure Ollama is running."

        if not response.message.tool_calls:
            return response.message.content

        for tool_call in response.message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            print(f"[Agent calling tool: {tool_name}]")

            if tool_name == "search_documents":
                result = search_documents(tool_args["query"], collection)
            elif tool_name == "search_web":
                result = search_web(tool_args["query"])
            else:
                result = "Tool not found."

            messages.append(response.message)
            messages.append({"role": "tool", "content": result})
    return "Sorry, I was unable to find an satisfactory answer after multiple attempts. Please try rephrasing your question."
