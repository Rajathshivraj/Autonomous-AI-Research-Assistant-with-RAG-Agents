from typing import List, Dict
from embeddings.embedder import Embedder
from vectorstore.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    """
    Coordinates the retrieval of relevant documents for a given query.
    Combines the Embedder and VectorStore.
    """
    
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieves the top-k most relevant documents for a query.
        
        Args:
            query (str): The user's question.
            k (int): Number of documents to retrieve.
            
        Returns:
            List[Dict]: List of retrieved documents with metadata.
            
        Flow:
        1. Convert query to embedding vector.
        2. Search vector store using that vector.
        3. Return results.
        """
        logger.info(f"Retrieving top {k} documents for query: '{query}'")
        
        # 1. Generate query embedding
        query_embedding = self.embedder.get_query_embedding(query)
        
        # 2. Search vector store
        results = self.vector_store.search_similarity(query_embedding, k=k)
        
        logger.info(f"Retrieved {len(results)} documents.")
        return results
