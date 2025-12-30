from pydantic import BaseModel
from typing import List, Dict, Optional

class QueryRequest(BaseModel):
    """
    Request model for the query endpoint.
    """
    question: str
    use_agent: bool = False  # Toggle between simple RAG and Agentic RAG

class Source(BaseModel):
    source: str
    text_snippet: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    """
    Response model ensuring consistent API output.
    """
    answer: str
    sources: List[Source]
    processing_time: float
    
class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    errors: Optional[List[str]] = None
