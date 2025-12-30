import chromadb
import uuid
import numpy as np
from typing import List, Dict, Any
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """
    Wrapper around ChromaDB for storing and retrieving vector embeddings.
    
    Why ChromaDB?
    - Open-source, easy to set up locally (just a file/folder).
    - Good performance for small to medium datasets.
    """
    
    def __init__(self, collection_name: str = "research_papers"):
        self.client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        logger.info(f"VectorStore initialized. Collection: {collection_name}, Path: {Config.VECTOR_DB_PATH}")

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None):
        """
        Adds text documents and their embeddings to the vector store.
        
        Args:
            documents (List[str]): List of text chunks.
            embeddings (List[List[float]]): List of embedding vectors.
            metadatas (List[Dict]): Optional metadata for each chunk (e.g., source file, page number).
            
        Why UUIDs?
        - Each document needs a unique ID in the vector store. 
        - UUID4 ensures global uniqueness without coordination.
        """
        count = len(documents)
        logger.info(f"Adding {count} documents to vector store...")
        
        if count == 0:
            logger.warning("No documents to add.")
            return

        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in range(count)]
        
        # Ensure metadata is provided for all docs
        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in range(count)]
            
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {count} documents.")
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise e

    def search_similarity(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Performs a semantic search using the query embedding.
        
        Args:
            query_embedding (List[float]): The vector representation of the query.
            k (int): Number of top results to return.
            
        Returns:
            List[Dict]: List of results with text, metadata, and distance/score.
        """
        logger.info(f"Performing similarity search for top {k} results...")
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Formatted results
            formatted_results = []
            
            # Chroma returns lists of lists (one list per query)
            # Since we query one by one, we access [0]
            if not results['documents']:
                return []
                
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                dist = results['distances'][0][i] if results['distances'] else 0.0
                
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "distance": dist, # Lower is better for cosine distance in Chroma (usually 1 - similarity)
                    "score": 1 - dist # Approximate similarity score
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise e
            
    def clear(self):
        """Clears all data in the collection. Useful for testing."""
        logger.warning(f"Clearing collection {self.collection.name}...")
        self.client.delete_collection(self.collection.name)
        # Re-create
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
