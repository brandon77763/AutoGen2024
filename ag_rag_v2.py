import gradio as gr
import os
import faiss
import numpy as np
import traceback
from openai import OpenAI
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb import PersistentClient

# Point to the local server
client = OpenAI(base_url="http://0.0.0.0:1234/v1", api_key="lm-studio")

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

# Initialize ChromaDB client
chroma_client = PersistentClient(path="/tmp/chromadb")
collection_name = "documents"
if collection_name not in [coll.name for coll in chroma_client.list_collections()]:
    chroma_client.create_collection(collection_name, metadata={"description": "Document storage"})

collection = chroma_client.get_collection(collection_name)

# Function to embed and index documents in FAISS and ChromaDB
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

    # Store in ChromaDB
    collection.add(doc_id, embedding_np.tolist())

    # Print statements for debugging
    print(f"Indexed document ID: {doc_id}")
    print(f"Content: {content[:100]}...")  # print the first 100 characters of the content
    print(f"Embedding: {embedding_np[:10]}...")  # print the first 10 elements of the embedding

    return f"File indexed successfully! Indexed document ID: {doc_id}"

# Function to query FAISS and ChromaDB
def query_faiss(query, session_state):
    query_embedding = embedding_generator.generate(query)
    query_embedding_np = np.array(query_embedding).astype('float32')
    D, I = index.search(np.array([query_embedding_np]), k=5)  # searching top 5
    results = []
    for idx in I[0]:
        if 0 <= idx < len(list(session_state['doc_id_to_embedding'].keys())):
            doc_id = list(session_state['doc_id_to_embedding'].keys())[idx]
            doc_metadata = collection.get(doc_id)
            results.append({
                "doc_id": doc_id,
                "content": session_state['doc_id_to_content'][doc_id]
            })
    print(f"Query results: {results}")  # Print query results for debugging
    return results

# Function to extract the most relevant information from the documents
def extract_relevant_info(documents, query):
    relevant_info = ""
    for doc in documents:
        # For simplicity, let's assume we just take the first few lines of each document
        # You can implement more sophisticated text extraction or summarization techniques here
        relevant_info += doc['content'][:500] + "\n"  # Get the first 500 characters of each doc
    return relevant_info

# Function to handle chat interaction
def chat_with_bot(user_message, chat_history, session_state):
    try:
        print(f"User message: {user_message}")

        # Query FAISS for relevant information
        faiss_results = query_faiss(user_message, session_state)
        context = ""
        if faiss_results:
            context = extract_relevant_info(faiss_results, user_message)

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

# Function to clear the ChromaDB collection and FAISS index
def clear_database():
    chroma_client.delete_collection(collection_name)
    chroma_client.create_collection(collection_name, metadata={"description": "Document storage"})
    global collection
    collection = chroma_client.get_collection(collection_name)  # reinitialize the collection
    global index
    index = faiss.IndexFlatL2(dimension)  # reinitialize the FAISS index
    return "Database cleared successfully!"

# Integrate AutoGen RAG
class CustomRetrieveAssistantAgent(RetrieveAssistantAgent):
    def __init__(self, *args, client=None, **kwargs):
        self.client = client
        super().__init__(*args, **kwargs)

    def get_llm_response(self, messages):
        completion = self.client.chat.completions.create(
            model=self.llm_config["model"],
            messages=messages,
            temperature=0.7,
        )
        return completion.choices[0].message.content

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "model": "microsoft/Phi-3-mini-4k-instruct-gguf"
}

assistant = CustomRetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    client=client,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": [],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 2000,
        "model": llm_config["model"],
        "client": chroma_client,
        "embedding_model": "all-mpnet-base-v2",
        "get_or_create": True,  
    },
    code_execution_config=False
)

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

        with gr.Column():
            clear_db_button = gr.Button("Clear Database")
            clear_db_output = gr.Textbox(label="Database status")

            clear_db_button.click(clear_database, inputs=[], outputs=clear_db_output)

iface.launch()
