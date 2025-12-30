from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from agents.tools import get_retrieval_tool
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class PlannerAgent:
    """
    An agent capable of multi-step reasoning. 
    It can plan, retrieve information, and synthesize an answer.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL_NAME, 
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=0  # Deterministic
        )
        self.tools = [get_retrieval_tool()]
        
        # Load the ReAct prompt from LangHub or define it. 
        # Using a standard prompt for robust instruction following.
        # If hub pulls fail, we can use a local prompt string.
        # self.prompt = hub.pull("hwchase17/react") 
        # For a truly offline/custom experience, I'll define it here.
        from langchain.prompts import PromptTemplate
        
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        self.prompt = PromptTemplate.from_template(template)
        
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, 
            handle_parsing_errors=True
        )

    def run(self, query: str) -> str:
        """
        Executes the agent workflow for a given query.
        """
        logger.info(f"Agent starting planning for: {query}")
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            return "I encountered an error while trying to solve this request."
