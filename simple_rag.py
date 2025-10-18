#!/usr/bin/env python3
"""
Minimal RAG System using Ollama all-minilm + Qdrant
Lightweight implementation for PDF processing and storage
"""

import os
import json
import logging
import requests
from typing import List, Dict, Any
from pathlib import Path
import hashlib
import re

# Minimal libraries
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import PyPDF2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    """
    Minimal Ollama embeddings client for all-minilm model
    """
    
    def __init__(self, model_name: str = "all-minilm", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.embed_url = f"{base_url}/api/embeddings"
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embeddings for a single text"""
        try:
            response = requests.post(
                self.embed_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = np.array(result["embedding"])
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([])
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings"""
        try:
            test_embedding = self.encode("test")
            return len(test_embedding) if len(test_embedding) > 0 else 384
        except:
            return 384

class MinimalRAG:
    """
    Minimal RAG system using Ollama + Qdrant
    """
    
    def __init__(self):
        """Initialize the RAG system"""
        # Configuration from environment
        self.collection_name = os.getenv("COLLECTION_NAME", "documents")
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
        
        # Initialize components
        logger.info("Initializing Ollama embeddings...")
        self.embedding_model = OllamaEmbeddings(base_url=self.ollama_url)
        self.embedding_dim = self.embedding_model.get_dimension()
        
        logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Qdrant collection with cosine similarity"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name} with dimension {self.embedding_dim}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_id(self, text: str, source: str, chunk_idx: int) -> str:
        """Generate unique ID for a chunk"""
        content = f"{source}_{chunk_idx}_{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_pdf(self, pdf_path: str) -> bool:
        """
        Add a PDF document to the vector store
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF not found: {pdf_path}")
                return False
            
            if pdf_path.suffix.lower() != '.pdf':
                logger.error(f"File is not a PDF: {pdf_path}")
                return False
            
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(str(pdf_path))
            
            if not text.strip():
                logger.warning(f"No text extracted from {pdf_path.name}")
                return False
            
            # Chunk the text
            chunks = self._chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
            
            # Generate embeddings and store
            points = []
            for idx, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {idx + 1}/{len(chunks)}")
                
                # Generate embedding using Ollama
                embedding = self.embedding_model.encode(chunk)
                
                if len(embedding) == 0:
                    logger.warning(f"Failed to generate embedding for chunk {idx}")
                    continue
                
                # Create unique ID
                point_id = self._generate_id(chunk, pdf_path.name, idx)
                
                # Create point with metadata
                point = models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk,
                        "source": pdf_path.name,
                        "chunk_index": idx,
                        "file_path": str(pdf_path)
                    }
                )
                points.append(point)
            
            if not points:
                logger.error("No valid embeddings generated")
                return False
            
            # Batch insert to Qdrant
            logger.info(f"Storing {len(points)} chunks in Qdrant...")
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"✅ Successfully added {len(points)} chunks from {pdf_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding PDF {pdf_path}: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            # Generate query embedding using Ollama
            query_embedding = self.embedding_model.encode(query)
            
            if len(query_embedding) == 0:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=0.1
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "text": result.payload["text"],
                    "source": result.payload["source"],
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value
            }
        except Exception as e:
            return {"error": str(e)}

def check_services():
    """Check if required services are running"""
    services = {}
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        services["ollama"] = response.status_code == 200
    except:
        services["ollama"] = False
    
    # Check Qdrant
    try:
        client = QdrantClient(url="http://localhost:6333")
        client.get_collections()
        services["qdrant"] = True
    except:
        services["qdrant"] = False
    
    return services

def main():
    """Main function for PDF processing"""
    
    print("🚀 Minimal RAG System with Ollama + Qdrant")
    print("=" * 50)
    
    # Check services
    print("🔍 Checking services...")
    services = check_services()
    
    if not services["ollama"]:
        print("❌ Ollama not running. Start with: ollama serve")
        print("💡 Make sure to pull the model: ollama pull all-minilm")
        return
    
    if not services["qdrant"]:
        print("❌ Qdrant not running. Start with: docker run -p 6333:6333 qdrant/qdrant")
        return
    
    print("✅ All services are running!")
    
    # Initialize RAG system
    try:
        rag = MinimalRAG()
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Interactive menu
    while True:
        print("\n" + "=" * 50)
        print("📋 Choose an option:")
        print("1. Add PDF to collection")
        print("2. Search documents")
        print("3. Collection statistics")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            pdf_path = input("Enter PDF file path: ").strip()
            if pdf_path:
                success = rag.add_pdf(pdf_path)
                if success:
                    print("✅ PDF added successfully!")
                else:
                    print("❌ Failed to add PDF")
        
        elif choice == "2":
            query = input("Enter your search query: ").strip()
            if query:
                print(f"\n🔍 Searching for: '{query}'")
                results = rag.search(query, top_k=3)
                
                if results:
                    print(f"\n📋 Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n--- Result {i} ---")
                        print(f"📄 Source: {result['source']}")
                        print(f"⭐ Score: {result['score']:.4f}")
                        print(f"📝 Text: {result['text'][:200]}...")
                else:
                    print("❌ No results found")
        
        elif choice == "3":
            stats = rag.get_stats()
            if "error" not in stats:
                print(f"\n📊 Collection Statistics:")
                print(f"  📄 Documents: {stats['points_count']}")
                print(f"  🔢 Vectors: {stats['vectors_count']}")
                print(f"  ✅ Status: {stats['status']}")
            else:
                print(f"❌ Error getting stats: {stats['error']}")
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()