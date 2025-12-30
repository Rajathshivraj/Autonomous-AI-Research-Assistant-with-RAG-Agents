from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class Generator:
    """
    Handles the interaction with the LLM to generate answers.
    """
    
    def __init__(self):
        # Initialize LangChain ChatModel
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME,
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=0.0 # Low temperature for factual consistency
        )

    def generate_answer(self, prompt: str) -> str:
        """
        Sends the prompt to the LLM and gets the response.
        
        Args:
            prompt (str): The full input prompt.
            
        Returns:
            str: The generated answer.
        """
        logger.info("Sending prompt to LLM...")
        try:
            # We can send it as a single HumanMsg or split it. 
            # Since our prompt builder already formatted it, we send it as one block.
            messages = [HumanMessage(content=prompt)]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            logger.info("Received answer from LLM.")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."
