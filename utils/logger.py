import logging
import sys
from logging.handlers import RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger instance.
    
    Args:
        name (str): The name of the logger, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: A configured logger instance.
        
    Why this is important:
    - Consistent logging format across all modules.
    - Captures both stdout (for development) and file logs (for auditing).
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicate logs if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 1. Stream Handler (Console Output)
    # Useful for seeing logs in real-time during development and debugging.
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    
    # 2. File Handler
    # detailed logs are saved to a file. RotatingFileHandler prevents the file from growing indefinitely.
    f_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    
    return logger
