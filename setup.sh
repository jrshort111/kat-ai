#!/bin/bash

echo "🐱 Setting up Kat AI Assistant..."

# Update system
echo "📦 Updating system packages..."
sudo apt update

# Install Python and pip if not already installed
echo "🐍 Installing Python dependencies..."
sudo apt install -y python3 python3-pip python3-venv

# Create project directory
echo "📁 Creating project directory..."
mkdir -p ~/kat-ai
cd ~/kat-ai

# Create virtual environment
echo "🏗️ Creating Python virtual environment..."
python3 -m venv kat-env
source kat-env/bin/activate

# Install Python packages
echo "📚 Installing Python packages..."
pip install streamlit chromadb requests

# Install Ollama
echo "🦙 Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
echo "🚀 Starting Ollama..."
ollama serve &
sleep 5

# Pull a default model
echo "📥 Downloading Llama 3.2 model (this may take a while)..."
ollama pull llama3.2

# Pull embedding model
echo "📥 Downloading embedding model..."
ollama pull nomic-embed-text

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Export your ChatGPT conversations:"
echo "   - Go to https://chat.openai.com"
echo "   - Settings → Data Export → Export"
echo "   - Download the conversations.json file"
echo ""
echo "2. Run Kat AI:"
echo "   cd ~/kat-ai"
echo "   source kat-env/bin/activate"
echo "   streamlit run kat_ai.py"
echo ""
echo "3. Open your browser to http://localhost:8501"
echo "4. Upload your conversations.json file in the sidebar"
echo "5. Start chatting with Kat!"
