from langchain_core.tools import Tool
from rag.retriever import Retriever
from utils.logger import get_logger

logger = get_logger(__name__)

def get_retrieval_tool() -> Tool:
    """
    Wraps the Retriever in a LangChain Tool for use by an agent.
    
    Returns:
        Tool: A tool that the agent can invoke.
    """
    retriever = Retriever()
    
    def retrieve_func(query: str) -> str:
        """
        Retrieves documents and returns them as a single string string.
        Agents consume text, so we flatten the list of dicts.
        """
        logger.info(f"Agent tool invoked: searching for '{query}'")
        results = retriever.retrieve(query)
        
        if not results:
            return "No relevant documents found."
            
        # Format results for the agent
        formatted_text = ""
        for i, res in enumerate(results):
            formatted_text += f"---\nSource: {res['metadata'].get('source', 'unknown')}\n{res['text']}\n"
            
        return formatted_text

    return Tool(
        name="Research_Retriever",
        func=retrieve_func,
        description="Useful for retrieving detailed information from the research papers. Input should be a specific search query."
    )
