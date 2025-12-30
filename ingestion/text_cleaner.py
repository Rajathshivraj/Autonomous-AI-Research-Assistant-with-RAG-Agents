import re
from utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDFs.
    
    Args:
        text (str): The raw text to clean.
        
    Returns:
        str: The cleaned text.
        
    Why this is needed:
    - PDFs often contain weird artifacts, excessive whitespace, or broken lines.
    - standardizing text improves embedding quality and retrieval accuracy.
    """
    logger.info("Cleaning extracted text...")
    
    # 1. Replace multiple newlines with a single newline (preserves paragraph structure somewhat)
    # But often in RAG, we want to remove soft line breaks within paragraphs.
    # Approach: Replace single newlines with space, keep double newlines as paragraph separators.
    
    # Normalize whitespace: replace multiple spaces/tabs with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    
    # (Optional) Fix hyphenation at line breaks (e.g., "communi- cation" -> "communication")
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    logger.info(f"Text cleaning complete. Final length: {len(text)}")
    return text.strip()
