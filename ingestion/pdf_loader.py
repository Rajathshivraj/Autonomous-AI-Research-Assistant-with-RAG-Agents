import os
from pypdf import PdfReader
from utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and extracts text from all pages.
    
    Args:
        file_path (str): The absolute path to the PDF file.
        
    Returns:
        str: The extracted text content from the PDF.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If there is an error reading the PDF.
        
    Why this implementation:
    - Uses `pypdf` which is a robust, pure-python library.
    - Iterates through pages to reconstruct the full document text.
    - Includes basic error handling to ensure pipeline resilience.
    """
    logger.info(f"Starting to load PDF from: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    try:
        reader = PdfReader(file_path)
        text = ""
        
        # Iterate over each page and extract text
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
        logger.info(f"Successfully loaded PDF. Total key characters extracted: {len(text)}")
        return text
        
    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        raise e
