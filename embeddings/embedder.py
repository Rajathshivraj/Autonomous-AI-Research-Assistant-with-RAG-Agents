from typing import List, Union
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class Embedder:
    """
    Handles generation of embeddings for text chunks.
    Supports both local models (SentenceTransformers) and API-based models (OpenAI).
    """

    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.api_key = Config.OPENAI_API_KEY
        self.model = self._load_model()

    def _load_model(self):
        """
        Loads the embedding model based on configuration.
        """
        logger.info(f"Loading embedding model: {self.model_name}")
        try:
            if "gpt" in self.model_name or "text-embedding" in self.model_name:
                # Use OpenAI Embeddings
                if not self.api_key:
                    raise ValueError("OpenAI API Key required for OpenAI embeddings.")
                return OpenAIEmbeddings(model=self.model_name, openai_api_key=self.api_key)
            else:
                # Default to SentenceTransformers (HuggingFace)
                # This downloads the model locally (first time)
                return SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise e

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text strings.
        
        Args:
            texts (List[str]): List of clean text chunks.
            
        Returns:
            List[List[float]]: List of vector embeddings.
            
        Why batches?
        - Processing in batches is significantly faster than one by one.
        """
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        try:
            if isinstance(self.model, SentenceTransformer):
                # SentenceTransformers returns numpy array, convert to list
                embeddings = self.model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()
            else:
                # OpenAI Embeddings via LangChain
                return self.model.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Generates embedding for a single query string.
        Critically important to use the SAME model for query and documents.
        """
        try:
            if isinstance(self.model, SentenceTransformer):
                embedding = self.model.encode(query, convert_to_tensor=False)
                return embedding.tolist()
            else:
                return self.model.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise e
