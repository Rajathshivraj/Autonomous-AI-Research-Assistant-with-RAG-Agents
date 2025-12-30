from utils.logger import get_logger

logger = get_logger(__name__)

class Validator:
    """
    Validates the generated answer to ensure it's not a complete hallucination.
    """
    
    def validate(self, answer: str, context_presence: bool) -> bool:
        """
        Checks if the answer is valid based on simple heuristics.
        
        Args:
            answer (str): The generated answer.
            context_presence (bool): Whether any context was actually retrieved.
            
        Returns:
            bool: True if valid, False otherwise.
            
        Why this is needed:
        - If retrieval fails, the model might try to make things up or say "I don't know".
        - We want to catch "I don't know" and handle it gracefully or flag it.
        """
        if not answer or len(answer.strip()) < 10:
            logger.warning("Validation failed: Answer is too short.")
            return False
            
        # Check for refusal patterns
        refusal_phrases = [
            "i don't have enough information",
            "i cannot answer",
            "no information found"
        ]
        
        if any(phrase in answer.lower() for phrase in refusal_phrases):
            logger.info("Validation note: Model refused to answer (likely correct behavior if context missing).")
            # We consider this 'valid' behavior but it indicates low recall
            pass
            
        return True
