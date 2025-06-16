#!/usr/bin/env python3
"""
Kat AI Assistant - Local RAG-based AI trained on your ChatGPT conversations
"""

import os
import json
import streamlit as st
import chromadb
from chromadb.config import Settings
import requests
import time
from typing import List, Dict, Any
import hashlib

# Configuration
OLLAMA_URL = "http://localhost:11434"
CHROMA_DB_PATH = "./chroma_db"
CONVERSATIONS_PATH = "./conversations"

class KatAI:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = None
        self.setup_database()

    def setup_database(self):
        """Initialize or get the ChromaDB collection"""
        try:
            self.collection = self.chroma_client.get_collection("kat_conversations")
            st.success("Found existing conversation database")
        except:
            self.collection = self.chroma_client.create_collection("kat_conversations")
            st.info("Created new conversation database")

    def check_ollama_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags")
            return response.status_code == 200
        except:
            return False

    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except:
            return []

    def pull_model(self, model_name):
        """Pull a model from Ollama"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/pull",
                json={"name": model_name},
                stream=True
            )
            return response.status_code == 200
        except:
            return False

    def embed_text(self, text: str, model="nomic-embed-text"):
        """Generate embeddings using Ollama"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            if response.status_code == 200:
                return response.json()["embedding"]
            return None
        except:
            return None

    def load_conversations_from_json(self, file_path: str):
        """Load conversations from ChatGPT export JSON - handles multiple formats"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            conversations = []

            # Handle different export formats
            if isinstance(data, list):
                # Format 1: List of conversations
                for conversation in data:
                    parsed_conv = self._parse_conversation(conversation)
                    if parsed_conv:
                        conversations.append(parsed_conv)

            elif isinstance(data, dict):
                # Format 2: Single conversation or wrapper object
                if 'conversations' in data:
                    # Nested conversations
                    for conversation in data['conversations']:
                        parsed_conv = self._parse_conversation(conversation)
                        if parsed_conv:
                            conversations.append(parsed_conv)
                else:
                    # Single conversation
                    parsed_conv = self._parse_conversation(data)
                    if parsed_conv:
                        conversations.append(parsed_conv)

            return conversations
        except Exception as e:
            st.error(f"Error loading conversations: {e}")
            st.error("Please check the file format and try again.")
            return []

    def _parse_conversation(self, conversation):
        """Parse a single conversation from various formats"""
        try:
            messages = []
            conv_id = conversation.get('id', str(hash(str(conversation))))
            title = conversation.get('title', 'Untitled Conversation')

            # Method 1: mapping format (newer exports)
            if 'mapping' in conversation:
                mapping = conversation['mapping']
                # Sort by create_time if available
                sorted_nodes = []
                for node_id, node in mapping.items():
                    if node and node.get('message'):
                        sorted_nodes.append(node)

                # Sort by create_time or keep original order
                if sorted_nodes and 'create_time' in sorted_nodes[0]:
                    sorted_nodes.sort(key=lambda x: x.get('create_time', 0))

                for node in sorted_nodes:
                    message = node.get('message')
                    if message and message.get('content'):
                        role = message.get('author', {}).get('role', 'unknown')
                        content = message.get('content')

                        if isinstance(content, dict) and 'parts' in content:
                            # Content with parts
                            parts = content['parts']
                            if parts and isinstance(parts, list):
                                text_parts = []
                                for part in parts:
                                    if isinstance(part, str):
                                        text_parts.append(part)
                                    elif isinstance(part, dict) and 'text' in part:
                                        text_parts.append(part['text'])
                                if text_parts:
                                    text = ' '.join(text_parts)
                                    messages.append(f"{role}: {text}")
                        elif isinstance(content, str):
                            # Direct string content
                            messages.append(f"{role}: {content}")

            # Method 2: direct messages format (older exports)
            elif 'messages' in conversation:
                for message in conversation['messages']:
                    role = message.get('role', message.get('from', 'unknown'))
                    content = message.get('content', message.get('text', ''))
                    if content:
                        messages.append(f"{role}: {content}")

            # Method 3: conversation array format
            elif isinstance(conversation, list):
                for i, message in enumerate(conversation):
                    role = 'user' if i % 2 == 0 else 'assistant'
                    if isinstance(message, str):
                        messages.append(f"{role}: {message}")
                    elif isinstance(message, dict):
                        content = message.get('content', message.get('text', ''))
                        if content:
                            messages.append(f"{role}: {content}")

            if messages:
                return {
                    'id': conv_id,
                    'title': title,
                    'content': '\n'.join(messages)
                }

            return None

        except Exception as e:
            st.warning(f"Skipped one conversation due to parsing error: {e}")
            return None

    def process_and_store_conversations(self, conversations: List[Dict]):
        """Process conversations and store in ChromaDB"""
        if not conversations:
            return

        # Check if conversations already exist
        existing_count = self.collection.count()
        if existing_count > 0:
            if st.button("Clear existing data and reload?"):
                self.collection.delete()
                self.collection = self.chroma_client.create_collection("kat_conversations")
            else:
                st.info(f"Database already has {existing_count} conversations. Click above to reload.")
                return

        progress_bar = st.progress(0)
        status_text = st.empty()

        documents = []
        metadatas = []
        ids = []

        for i, conv in enumerate(conversations):
            # Split long conversations into chunks
            content = conv['content']
            chunks = self.split_text(content, max_length=1000)

            for j, chunk in enumerate(chunks):
                doc_id = f"{conv['id']}_chunk_{j}"
                documents.append(chunk)
                metadatas.append({
                    'conversation_id': conv['id'],
                    'title': conv['title'],
                    'chunk_index': j
                })
                ids.append(doc_id)

            progress_bar.progress((i + 1) / len(conversations))
            status_text.text(f"Processing conversation {i + 1}/{len(conversations)}")

        # Store in ChromaDB
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            st.success(f"Processed and stored {len(documents)} conversation chunks!")

    def split_text(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def search_conversations(self, query: str, n_results: int = 3):
        """Search for relevant conversations"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            st.error(f"Search error: {e}")
            return None

    def generate_response(self, query: str, context: str, model: str = "llama3.2"):
        """Generate response using Ollama"""
        prompt = f"""You are Kat, an AI assistant trained on the user's previous conversations. Use the following context from their past conversations to inform your response, but respond naturally as if you're continuing an ongoing conversation with them.

Context from past conversations:
{context}

Current question: {query}

Response:"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Sorry, I had trouble generating a response."
        except Exception as e:
            return f"Error: {e}"

    def chat(self, user_input: str, model: str = "llama3.2"):
        """Main chat function"""
        # Search for relevant context
        search_results = self.search_conversations(user_input)

        context = ""
        if search_results and search_results['documents']:
            context = "\n\n".join(search_results['documents'][0][:3])

        # Generate response
        response = self.generate_response(user_input, context, model)
        return response, context

def main():
    st.set_page_config(page_title="Kat AI Assistant", page_icon="🐱", layout="wide")

    st.title("🐱 Kat AI Assistant")
    st.markdown("*Your personal AI trained on your ChatGPT conversations*")

    # Initialize Kat
    if 'kat' not in st.session_state:
        st.session_state.kat = KatAI()

    kat = st.session_state.kat

    # Sidebar for setup
    with st.sidebar:
        st.header("🔧 Setup")

        # Check Ollama connection
        if kat.check_ollama_connection():
            st.success("✅ Ollama connected")

            # Model selection
            available_models = kat.get_available_models()
            if available_models:
                selected_model = st.selectbox("Select Model:", available_models)
            else:
                st.warning("No models found. Install a model first:")
                st.code("ollama pull llama3.2")
                selected_model = "llama3.2"

        else:
            st.error("❌ Ollama not running")
            st.markdown("Start Ollama first:")
            st.code("ollama serve")
            return

        st.divider()

        # Data loading section
        st.header("📚 Load Conversations")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload ChatGPT conversations (JSON)",
            type=['json'],
            help="Export your ChatGPT conversations and upload the JSON file"
        )

        if uploaded_file:
            # Save uploaded file
            os.makedirs(CONVERSATIONS_PATH, exist_ok=True)
            file_path = os.path.join(CONVERSATIONS_PATH, uploaded_file.name)

            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Debug: Show file structure
            with st.expander("🔍 Debug: File Structure"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    st.write("**File type:**", type(data).__name__)
                    if isinstance(data, list):
                        st.write("**Number of items:**", len(data))
                        if data:
                            st.write("**First item structure:**")
                            st.json(list(data[0].keys()) if isinstance(data[0], dict) else str(type(data[0])))

                            # Show first few keys of first conversation
                            if isinstance(data[0], dict):
                                st.write("**Sample data from first conversation:**")
                                sample = {}
                                for key in list(data[0].keys())[:5]:  # First 5 keys
                                    value = data[0][key]
                                    if isinstance(value, (str, int, float, bool)):
                                        sample[key] = value
                                    elif isinstance(value, list):
                                        sample[key] = f"[List with {len(value)} items]"
                                    elif isinstance(value, dict):
                                        sample[key] = f"{{Dict with keys: {list(value.keys())[:3]}}}"
                                    else:
                                        sample[key] = str(type(value))
                                st.json(sample)

                    elif isinstance(data, dict):
                        st.write("**Keys in root object:**", list(data.keys()))

                except Exception as e:
                    st.error(f"Debug error: {e}")

            # Load and process conversations
            conversations = kat.load_conversations_from_json(file_path)
            st.info(f"Found {len(conversations)} conversations")

            # Show sample conversation if available
            if conversations:
                with st.expander("📝 Sample Conversation"):
                    sample_conv = conversations[0]
                    st.write(f"**Title:** {sample_conv['title']}")
                    st.write(f"**Content preview:**")
                    preview = sample_conv['content'][:500] + "..." if len(sample_conv['content']) > 500 else sample_conv['content']
                    st.text(preview)

            if st.button("Process Conversations"):
                kat.process_and_store_conversations(conversations)

        # Database status
        if kat.collection:
            count = kat.collection.count()
            st.info(f"Database: {count} conversation chunks")

    # Main chat interface
    st.header("💬 Chat with Kat")

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, context = kat.chat(prompt, selected_model if 'selected_model' in locals() else "llama3.2")

            st.markdown(response)

            # Show context used (optional)
            if context and st.checkbox("Show context used", key=f"context_{len(st.session_state.messages)}"):
                with st.expander("Context from your conversations:"):
                    st.text(context[:500] + "..." if len(context) > 500 else context)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
