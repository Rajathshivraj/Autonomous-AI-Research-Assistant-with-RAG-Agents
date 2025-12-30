import time
import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List

from api.schemas import QueryRequest, QueryResponse, Source, IngestResponse
from ingestion.pdf_loader import load_pdf
from ingestion.text_cleaner import clean_text
from ingestion.chunker import chunk_text
from embeddings.embedder import Embedder
from vectorstore.vector_store import VectorStore
from rag.retriever import Retriever
from rag.prompt_builder import build_rag_prompt
from rag.generator import Generator
from agents.planner_agent import PlannerAgent
from utils.logger import get_logger

# Initialize API and Logger
app = FastAPI(title="Autonomous AI Research Assistant API")
logger = get_logger(__name__)

# Initialize components globally to avoid re-loading on every request
# In production, we might use dependency injection
try:
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever()
    generator = Generator()
    planner_agent = PlannerAgent()
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    # We continue, but endpoints might fail if not fixed.

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF, process it, and store embeddings.
    """
    logger.info(f"Received file upload: {file.filename}")
    
    # Save file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Load
        text = load_pdf(temp_path)
        
        # 2. Clean
        clean_txt = clean_text(text)
        
        # 3. Chunk
        chunks = chunk_text(clean_txt)
        
        # 4. Embed
        embeddings = embedder.get_embeddings(chunks)
        
        # 5. Store
        metadatas = [{"source": file.filename, "page": "unknown"} for _ in chunks]
        vector_store.add_documents(chunks, embeddings, metadatas)
        
        # Cleanup
        os.remove(temp_path)
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            chunks_added=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoint to ask a question to the system.
    Supports both standard RAG and Agentic workflows.
    """
    start_time = time.time()
    logger.info(f"Received query: {request.question}, Use Agent: {request.use_agent}")
    
    try:
        if request.use_agent:
            # Agentic Workflow
            answer = planner_agent.run(request.question)
            # Agents often absorb sources into the text, extracting them explicitly is harder 
            # without structured agent outputs. For now, we return empty sources or 
            # we could fetch relevant docs again to populate sources (hybrid approach).
            
            # Simple re-retrieval for citation (optional but good for UI)
            retrieved_docs = retriever.retrieve(request.question, k=3)
            sources = [
                Source(
                    source=doc['metadata'].get('source', 'unknown'),
                    text_snippet=doc['text'][:200] + "...",
                    score=doc.get('score')
                ) for doc in retrieved_docs
            ]
            
        else:
            # Standard RAG Workflow
            retrieved_docs = retriever.retrieve(request.question)
            prompt = build_rag_prompt(request.question, retrieved_docs)
            answer = generator.generate_answer(prompt)
            
            sources = [
                Source(
                    source=doc['metadata'].get('source', 'unknown'),
                    text_snippet=doc['text'][:200] + "...",
                    score=doc.get('score')
                ) for doc in retrieved_docs
            ]

        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
