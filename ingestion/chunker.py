from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import get_logger
from utils.config import Config

logger = get_logger(__name__)

def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
    """
    Splits text into smaller, overlapping chunks for embedding.
    
    Args:
        text (str): The collected text to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.
        
    Returns:
        List[str]: A list of text chunks.
        
    Why RecursiveCharacterTextSplitter:
    - It tries to split on paragraph separators (\n\n), then sentences (\n), then spaces.
    - This preserves semantic meaning better than fixed-size character splitting.
    - Overlap is critical to ensure context isn't lost at the boundaries of chunks.
    """
    logger.info(f"Chunking text with size={chunk_size} and overlap={chunk_overlap}...")
    
    if not text:
        logger.warning("Input text is empty. Returning empty chunk list.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_text(text)
    
    logger.info(f"Generated {len(chunks)} chunks.")
    return chunks
