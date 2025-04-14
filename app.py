import os
# import json
# import requests
# import asyncio
# from bs4 import BeautifulSoup
# import google.generativeai as genai
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO

# class RAGSystem:
#     def __init__(self, api_key):
#         # Configure the API key
#         genai.configure(api_key=api_key)
        
#         # Initialize the model with the latest SDK patterns
#         generation_config = {
#             "temperature": 0.2,
#             "top_p": 0.95,
#             "top_k": 40,
#             "max_output_tokens": 2048,
#         }
        
#         safety_settings = [
#             {
#                 "category": "HARM_CATEGORY_HARASSMENT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_HATE_SPEECH",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             },
#             {
#                 "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#             }
#         ]
        
#         self.model = genai.GenerativeModel(
#             model_name="gemini-1.5-flash",
#             generation_config=generation_config,
#             safety_settings=safety_settings
#         )
        
#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100,
#             length_function=len,
#         )
        
#         self.vector_store = None
#         self.ingested_urls = []
#         self.doc_count = 0
        
#         self.chat_histories = {}
    
#     def extract_text_from_url(self, url):
#         """Extract text content from a URL"""
#         try:
#             headers = {
#                 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#             }
#             response = requests.get(url, headers=headers, timeout=10)
#             response.raise_for_status()
            
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Remove script, style, and nav elements
#             for element in soup(['script', 'style', 'nav', 'footer', 'header']):
#                 element.extract()
            
#             # Get text
#             text = soup.get_text(separator=' ', strip=True)
            
#             # Clean up text (remove extra whitespace)
#             lines = (line.strip() for line in text.splitlines())
#             chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#             text = ' '.join(chunk for chunk in chunks if chunk)
            
#             return text
#         except Exception as e:
#             print(f"Error extracting text from {url}: {e}")
#             return ""
    
#     def ingest_urls(self, urls):
#         """Ingest content from URLs and build vector store"""
#         all_texts = []
#         successful_urls = []
        
#         for url in urls:
#             print(f"Ingesting content from: {url}")
#             text = self.extract_text_from_url(url)
#             if text:
#                 all_texts.append(text)
#                 successful_urls.append(url)
        
#         if not all_texts:
#             return False, "No content was successfully extracted.", 0, 0
        
#         # Split texts into chunks
#         docs = []
#         for i, text in enumerate(all_texts):
#             chunks = self.text_splitter.split_text(text)
#             for j, chunk in enumerate(chunks):
#                 docs.append({
#                     "id": f"doc_{i}_{j}",
#                     "content": chunk,
#                     "metadata": {"source": successful_urls[i] if i < len(successful_urls) else "unknown"}
#                 })
        
#         # Create vector store
#         texts = [doc["content"] for doc in docs]
#         metadatas = [{"source": doc["metadata"]["source"]} for doc in docs]
        
#         self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
#         self.ingested_urls = successful_urls
#         self.doc_count = len(docs)
        
#         return True, "Successfully ingested content.", len(docs), len(successful_urls)
    
#     def retrieve_relevant_context(self, query, top_k=5):
#         """Retrieve most relevant document chunks for a query"""
#         if not self.vector_store:
#             return []
        
#         results = self.vector_store.similarity_search(query, k=top_k)
#         return results
    
#     def get_chat_history(self, session_id):
#         """Get chat history for a specific session"""
#         if session_id not in self.chat_histories:
#             self.chat_histories[session_id] = []
#         return self.chat_histories[session_id]
    
#     def add_to_history(self, session_id, role, content):
#         """Add a message to the chat history"""
#         history = self.get_chat_history(session_id)
#         history.append({"role": role, "content": content})
#         # Limit history length to prevent token overflow
#         if len(history) > 20:
#             history.pop(0)
    
#     def generate_streaming_response(self, session_id, query, callback):
#         """Generate an answer using RAG with streaming response"""
#         if not self.vector_store:
#             callback({"chunk": "Please ingest URLs first before asking questions."})
#             return [], []
        
#         # Retrieve relevant context
#         relevant_docs = self.retrieve_relevant_context(query)
        
#         if not relevant_docs:
#             callback({"chunk": "No relevant information found to answer your question."})
#             return [], []
        
#         # Prepare context for Gemini
#         context = "\n\n".join([doc.page_content for doc in relevant_docs])
#         sources = [doc.metadata.get("source", "unknown") for doc in relevant_docs]
        
#         # Get chat history
#         history = self.get_chat_history(session_id)
#         history_text = ""
#         if history:
#             for msg in history[-5:]:  # Only use last 5 messages to keep prompt size reasonable
#                 history_text += f"{msg['role']}: {msg['content']}\n"
        
#         # Using the parts-based prompt approach
#         prompt_parts = [
#             "You are a helpful assistant that provides accurate information based ONLY on the provided context. If the information is not in the context, say 'I don't have enough information to answer this question.' Be concise and accurate.",
#             f"\nContext:\n{context}",
#             f"\nPrevious conversation (if any):\n{history_text}" if history_text else "",
#             f"\nUser Question: {query}",
#             "\nAnswer:"
#         ]
#         print("..................>>",prompt_parts)
#         # Generate streaming response
#         response_text = ""
#         try:
#             response = self.model.generate_content(prompt_parts)
#             print(response)
#             for chunk in response:
#                 if chunk.text:
#                     response_text += chunk.text
#                     callback({"chunk": chunk.text})
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             callback({"chunk": "An error occurred while generating the response. Please try again later."})

        
#         # Add the interaction to history
#         self.add_to_history(session_id, "user", query)
#         self.add_to_history(session_id, "assistant", response_text)
        
#         # Return the complete response and sources
#         unique_sources = list(set(sources))
#         return response_text, unique_sources

# # Initialize the RAG system with API key from environment variable
# api_key = os.environ.get("GOOGLE_API_KEY")
# if not api_key:
#     print("Warning: GEMINI_API_KEY environment variable not set.")
#     api_key = "DEMO_KEY"  # Will be replaced by user input in production
from core import RSYS
rag_system = RSYS()

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/api/status', methods=['GET'])
# def status():
#     return jsonify({
#         'ready': rag_system.vector_store is not None,
#         'doc_count': rag_system.doc_count,
#         'url_count': len(rag_system.ingested_urls),
#         'urls': rag_system.ingested_urls
#     })

@app.route('/api/ingest', methods=['POST'])
def ingest():
    urls = request.json.get('urls', [])
    
    if not urls:
        return jsonify({'success': False, 'message': 'No URLs provided'})
    rag_system.add_to_vdb(urls)
    # success, message, doc_count, url_count = rag_system.ingest_urls(urls)
    
    return jsonify({
        'success': True,
        'message': "Succesfull",
        'doc_count': doc_count,
        'url_count': len(urls)
    })

# WebSocket event handlers
@socketio.on('connect')
def handle_connect(data):
    print(f"Client connected: {data}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")


@socketio.on('user_message')
def handle_message(data):
    print("...................",data)

    message = data['message']
    session_id = request.sid
    
    if not message:
        emit('error', {'error': 'Message is empty'}, room=session_id)
        return
    print("...................",message)
    # Function to emit chunks to the client
    # def emit_chunk(chunk):
    #     emit('response_chunk', chunk, room=session_id)
    response = rag_system.main(message)
    # Generate streaming response
    # response, sources = rag_system.generate_streaming_response(session_id, message, emit_chunk)
    
    # Signal that the response is complete
    emit('response_complete', {'response': response, 'sources': 'sources'}, room=session_id)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
