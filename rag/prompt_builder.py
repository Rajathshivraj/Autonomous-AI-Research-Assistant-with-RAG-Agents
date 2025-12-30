from typing import List, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

def build_rag_prompt(query: str, context_chunks: List[Dict]) -> str:
    """
    Constructs the final prompt for the LLM by combining the query and context.
    
    Args:
        query (str): The user's original question.
        context_chunks (List[Dict]): List of retrieved documents.
        
    Returns:
        str: The full prompt string.
        
    Why this structure?
    - "System" instructions guide the LLM to only use the provided context.
    - This reduces hallucinations (making things up) by grounding the answer.
    """
    logger.info("Building RAG prompt...")
    
    # 1. Join context texts
    # We use a numbered list or simple separation to keep chunks distinct
    context_text = ""
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text", "")
        source = chunk.get("metadata", {}).get("source", "unknown")
        context_text += f"source: {source}\ncontent: {text}\n\n"
        
    # 2. Construct the prompt template
    prompt = f"""You are a helpful AI research assistant. Use the following context to answer the user's question.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up facts.

---
CONTEXT:
{context_text}
---

USER QUESTION:
{query}

ANSWER:
"""
    return prompt
