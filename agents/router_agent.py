"""Router agent that determines which agent should handle the request."""
from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings


class RouterAgent:
    """Routes requests to appropriate specialized agents."""
    
    def __init__(self):
        """Initialize router agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.1,
            api_key=settings.openai_api_key
        )
    
    def route(self, query: str, has_image: bool = False) -> Dict[str, Any]:
        """
        Determine which agent should handle the request.
        
        Args:
            query: User query
            has_image: Whether the request includes an image
            
        Returns:
            Dict with 'agent' (agent name) and 'reasoning' (explanation)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a routing agent. Analyze the user's query and determine 
            which specialized agent should handle it. Available agents:
            - text_reasoning: For complex reasoning, analysis, problem-solving
            - vision: For image analysis, OCR, visual understanding
            - retrieval: For questions requiring knowledge retrieval, RAG
            - tool: For tasks requiring external tools (web search, calculations, APIs)
            
            Respond with ONLY the agent name (one of: text_reasoning, vision, retrieval, tool)."""),
            ("user", "Query: {query}\nHas image: {has_image}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "has_image": has_image
        })
        
        agent_name = response.content.strip().lower()
        
        # Validate and default
        valid_agents = ["text_reasoning", "vision", "retrieval", "tool"]
        if agent_name not in valid_agents:
            # Default routing logic
            if has_image:
                agent_name = "vision"
            elif any(keyword in query.lower() for keyword in ["search", "find", "lookup", "current"]):
                agent_name = "tool"
            elif any(keyword in query.lower() for keyword in ["what", "who", "when", "where", "how", "explain"]):
                agent_name = "retrieval"
            else:
                agent_name = "text_reasoning"
        
        return {
            "agent": agent_name,
            "reasoning": f"Routed to {agent_name} based on query analysis"
        }

