# Simple RAG System with Qdrant

A minimal RAG (Retrieval-Augmented Generation) system implementation using Qdrant vector database and sentence transformers for embeddings.

## 🚀 Quick Start

### 1. Setup Environment (WSL)
```bash
# Create virtual environment
python3 -m venv rag-env
source rag-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant
```bash
# Run Qdrant with Docker
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 3. Run the RAG System
```bash
python simple_rag.py
```

## 📋 Features

- **Document Processing**: PDF and TXT file support
- **Vector Storage**: Qdrant with cosine similarity
- **Text Chunking**: Configurable chunk size and overlap
- **Similarity Search**: Fast cosine similarity search
- **Interactive CLI**: Easy-to-use command-line interface

## 🛠️ Components

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Qdrant with cosine distance
- **Document Types**: PDF, TXT
- **Text Processing**: Automatic chunking with overlap

## 📁 Usage

1. **Add Documents**: Upload PDF or TXT files
2. **Search**: Query your documents with natural language
3. **View Results**: Get ranked results with similarity scores
4. **Manage**: Clear collection or view statistics

## ⚙️ Configuration

Edit `.env` file to customize:
- Qdrant connection settings
- Embedding model selection
- Chunk size and overlap
- Collection name

Perfect for RAG demonstrations and prototyping!