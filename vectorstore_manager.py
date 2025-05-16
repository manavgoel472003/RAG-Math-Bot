import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuration
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 80

class VectorstoreManager:
    def __init__(self, store_path: str):
        self.store_path = Path(store_path)
        # Create directory if it doesn't exist
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.embedder = OllamaEmbeddings(base_url=OLLAMA_BASE, model=EMBED_MODEL)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True
        )
        self.vectorstore = self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self) -> FAISS:
        """Load existing vectorstore or create a new one"""
        index_path = self.store_path / "index.faiss"
        if index_path.exists():
            print(f"Loading existing vectorstore from {self.store_path}")
            return FAISS.load_local(
                self.store_path, 
                self.embedder,
                allow_dangerous_deserialization=True
            )
        else:
            print(f"Creating new vectorstore at {self.store_path}")
            return FAISS.from_texts(
                ["Initial document"], 
                self.embedder,
                distance_strategy="COSINE"
            )

    def load_document(self, file_path: Path) -> List[Document]:
        """Load a document based on its file type"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                return loader.load()
            elif suffix in {".txt", ".md"}:
                loader = TextLoader(str(file_path), encoding="utf-8")
                return loader.load()
            else:
                loader = UnstructuredFileLoader(str(file_path))
                return loader.load()
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            raise

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to the vectorstore"""
        # Split documents into chunks
        splits = self.splitter.split_documents(documents)
        
        # Add to vectorstore
        self.vectorstore.add_documents(splits)
        
        # Save the updated vectorstore
        self.save()

    def add_file(self, file_path: str) -> None:
        """Add a file to the vectorstore"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        documents = self.load_document(file_path)
        self.add_documents(documents)
        print(f"Added {file_path} to vectorstore")

    def save(self) -> None:
        """Save the vectorstore to disk"""
        self.vectorstore.save_local(self.store_path)
        print(f"Vectorstore saved to {self.store_path}")

    def get_vectorstore(self) -> FAISS:
        """Get the current vectorstore instance"""
        return self.vectorstore

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = VectorstoreManager("path/to/your/vectorstore")
    
    # Add a single file
    manager.add_file("path/to/your/document.pdf")
    
    # Add multiple files
    files_to_add = [
        "path/to/file1.txt",
        "path/to/file2.md",
        "path/to/file3.pdf"
    ]
    
    for file_path in files_to_add:
        try:
            manager.add_file(file_path)
        except Exception as e:
            print(f"Error adding {file_path}: {str(e)}")
    
    # Save the final vectorstore
    manager.save() 