import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to manage environment variables and settings.
    This centralized approach makes it easier to manage settings across the project.
    """
    
    # OpenAI API Key - Critical for access to GPT models
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Path to the persistent Vector Database
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    
    # Embedding model name - using a standard SentenceTransformer model
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # LLM Model Name
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

    @staticmethod
    def validate():
        """
        Validates specific critical configuration values.
        Raises an error if mandatory keys are missing.
        """
        if not Config.OPENAI_API_KEY:
            # We log a warning or error here, but for now just a print/pass or raise
            raise ValueError("OPENAI_API_KEY is missing. Please set it in the .env file.")

# Using a robust pattern to ensure config is valid at startup
# Config.validate() # simple check, can be called in main.py
