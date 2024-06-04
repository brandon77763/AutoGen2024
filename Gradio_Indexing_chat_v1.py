import gradio as gr
import os
import faiss
import numpy as np
import traceback
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://192.168.1.194:1234/v1", api_key="lm-studio")

# Function to generate embeddings using local LLM Studio
class LocalEmbeddingGenerator:
    def __init__(self, client):
        self.client = client

    def generate(self, text):
        response = self.client.embeddings.create(
            model="microsoft/Phi-3-mini-4k-instruct-gguf",
            input=text
        )
        return response.data[0].embedding

# Function to get chat completions using local LLM Studio
class LocalChatCompleter:
    def __init__(self, client):
        self.client = client

    def complete(self, messages):
        print(f"Sending request to chat completions API with messages: {messages}")  # Debugging info
        response = self.client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct-gguf",
            messages=messages,
            temperature=0.7,
        )
        print(f"Received response: {response}")  # Debugging info
        return response.choices[0].message.content

embedding_generator = LocalEmbeddingGenerator(client)
chat_completer = LocalChatCompleter(client)

# Initialize FAISS index
dimension = 768  # dimension of your embeddings
index = faiss.IndexFlatL2(dimension)  # using L2 (Euclidean) distance

# Function to embed and index documents
def embed_and_index(file, session_state):
    if file is None:
        return "No file uploaded"
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file.name, 'r', encoding='ISO-8859-1') as f:
            content = f.read()

    embedding = embedding_generator.generate(content)
    embedding_np = np.array(embedding).astype('float32')
    index.add(np.array([embedding_np]))  # adding to FAISS index
    doc_id = os.path.basename(file.name)
    session_state['doc_id_to_embedding'][doc_id] = embedding_np
    session_state['doc_id_to_content'][doc_id] = content
    
    # Print statements for debugging
    print(f"Indexed document ID: {doc_id}")
    print(f"Content: {content[:100]}...")  # print the first 100 characters of the content
    print(f"Embedding: {embedding_np[:10]}...")  # print the first 10 elements of the embedding

    return f"File indexed successfully! Indexed document ID: {doc_id}"

# Function to query FAISS
def query_faiss(query, session_state):
    query_embedding = embedding_generator.generate(query)
    query_embedding_np = np.array(query_embedding).astype('float32')
    D, I = index.search(np.array([query_embedding_np]), k=5)  # searching top 5
    results = []
    for idx in I[0]:
        if 0 <= idx < len(list(session_state['doc_id_to_embedding'].keys())):
            doc_id = list(session_state['doc_id_to_embedding'].keys())[idx]
            results.append({
                "doc_id": doc_id,
                "content": session_state['doc_id_to_content'][doc_id]
            })
    print(f"Query results: {results}")  # Print query results for debugging
    return results

# Function to handle chat interaction
def chat_with_bot(user_message, chat_history, session_state):
    try:
        print(f"User message: {user_message}")

        # Query FAISS for relevant information
        faiss_results = query_faiss(user_message, session_state)
        context = ""
        if faiss_results:
            context = "Here is the relevant information from the documents:\n"
            for result in faiss_results:
                context += f"- {result['content']}\n"

        # Integrate context into the user message
        if context:
            user_message += f"\n\n{context.strip()}"

        history = [{"role": "user", "content": user_message}]

        print(f"History before completion: {history}")  # Debugging info

        # Use the chat completion function to generate a response
        completion_response = chat_completer.complete(history)
        response = completion_response

        print(f"Bot response: {response}")
        chat_history.append((f"You: {user_message}", f"Bot: {response}"))
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        chat_history.append((f"Bot: {error_message}", ""))
    return chat_history

# Gradio interface for chat and file uploads
with gr.Blocks() as iface:
    gr.Markdown("# Chat with AI")
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            query_input = gr.Textbox(label="Enter your message")
            chat_button = gr.Button("Send")

            state = gr.State({'doc_id_to_embedding': {}, 'doc_id_to_content': {}})

            def submit_message(user_message, chat_history, session_state):
                new_history = chat_with_bot(user_message, chat_history, session_state)
                return "", new_history

            query_input.submit(submit_message, [query_input, chatbot, state], [query_input, chatbot])
            chat_button.click(submit_message, [query_input, chatbot, state], [query_input, chatbot])
        
        with gr.Column():
            file_input = gr.File()
            upload_button = gr.Button("Upload Document")
            upload_output = gr.Textbox(label="Upload status")

            def upload_document(file, session_state):
                return embed_and_index(file, session_state)

            upload_button.click(upload_document, inputs=[file_input, state], outputs=upload_output)

iface.launch()
