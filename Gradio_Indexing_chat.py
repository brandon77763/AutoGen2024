import gradio as gr
import os
import faiss
import numpy as np
import requests
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
        response = self.client.chat.completions.create(
            model="microsoft/Phi-3-mini-4k-instruct-gguf",
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

embedding_generator = LocalEmbeddingGenerator(client)
chat_completer = LocalChatCompleter(client)

# Initialize FAISS index
dimension = 768  # dimension of your embeddings
index = faiss.IndexFlatL2(dimension)  # using L2 (Euclidean) distance

# Storage for document IDs and their corresponding embeddings
doc_id_to_embedding = {}
doc_id_to_content = {}

# Function to embed and index documents
def embed_and_index(file):
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
    doc_id_to_embedding[doc_id] = embedding_np
    doc_id_to_content[doc_id] = content
    
    # Print statements for debugging
    print(f"Indexed document ID: {doc_id}")
    print(f"Content: {content[:100]}...")  # print the first 100 characters of the content
    print(f"Embedding: {embedding_np[:10]}...")  # print the first 10 elements of the embedding

    return f"File indexed successfully! Indexed document ID: {doc_id}"

# Function to query FAISS
def query_faiss(query):
    query_embedding = embedding_generator.generate(query)
    query_embedding_np = np.array(query_embedding).astype('float32')
    D, I = index.search(np.array([query_embedding_np]), k=5)  # searching top 5
    results = []
    for idx in I[0]:
        if idx != -1:
            doc_id = list(doc_id_to_embedding.keys())[idx]
            results.append({
                "doc_id": doc_id,
                "content": doc_id_to_content[doc_id]
            })
    print(f"Query results: {results}")  # Print query results for debugging
    return results

# Function to handle chat interaction
def chat_with_bot(user_message):
    history = [{"role": "user", "content": user_message}]
    try:
        print(f"User message: {user_message}")

        # Query FAISS for relevant information
        faiss_results = query_faiss(user_message)
        if faiss_results:
            response = f"I found the following information:\n"
            for result in faiss_results:
                response += f"Document ID: {result['doc_id']}\nContent: {result['content'][:200]}...\n\n"
        else:
            # Use the chat completion function to generate a response
            completion_response = chat_completer.complete(history)
            response = completion_response

        print(f"Bot response: {response}")
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        print(traceback.format_exc())
        history.append({"role": "assistant", "content": error_message})
    return history[-1]['content']

# Gradio interface for chat and file uploads
with gr.Blocks() as iface:
    gr.Markdown("# Chat with AI")
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(lines=2, placeholder="Enter your query here...")
            chat_button = gr.Button("Chat")
            chat_output = gr.Textbox(lines=10, placeholder="Chatbot response...")
            chat_button.click(chat_with_bot, inputs=query_input, outputs=chat_output)
        
        with gr.Column():
            file_input = gr.File()
            upload_button = gr.Button("Upload Document")
            upload_output = gr.Textbox(lines=2, placeholder="Upload status...")
            upload_button.click(embed_and_index, inputs=file_input, outputs=upload_output)

iface.launch()
