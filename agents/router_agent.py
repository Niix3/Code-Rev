"""Router agent that determines which agent should handle the request."""
from typing import Dict, Any
from openai import OpenAI
from config import settings


class RouterAgent:
    """Routes requests to appropriate specialized agents."""
    
    def __init__(self):
        """Initialize router agent."""
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Determine which agent should handle the request.
        
        Args:
            query: User query
            
        Returns:
            Dict with 'agent' (agent name) and 'reasoning' (explanation)
        """
        system_prompt = """You are a routing agent. Analyze the user's query and determine
which specialized agent should handle it. Available agents:
- text_reasoning: For complex reasoning, analysis, problem-solving
- tool: For tasks requiring external tools (web search, calculations, APIs)

Respond with ONLY the agent name (one of: text_reasoning, tool)."""

        completion = self.client.chat.completions.create(
            model=settings.default_llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"},
            ],
        )
        agent_name = (completion.choices[0].message.content or "").strip().lower()
        
        # Validate and default
        valid_agents = ["text_reasoning", "tool"]
        if agent_name not in valid_agents:
            # Default routing logic
            if any(keyword in query.lower() for keyword in ["search", "find", "lookup", "current"]):
                agent_name = "tool"
            else:
                agent_name = "text_reasoning"
        
        return {
            "agent": agent_name,
            "reasoning": f"Routed to {agent_name} based on query analysis"
        }

