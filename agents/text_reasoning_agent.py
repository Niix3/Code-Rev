"""Text reasoning agent for complex reasoning tasks."""
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings


class TextReasoningAgent:
    """Handles complex reasoning and analysis tasks."""
    
    def __init__(self):
        """Initialize text reasoning agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    
    def reason(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform reasoning on the query.
        
        Args:
            query: User query
            context: Optional context from other agents
            
        Returns:
            Dict with 'response' and 'reasoning_steps'
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert reasoning agent. Analyze the problem step by step,
            show your reasoning process, and provide a clear, well-reasoned answer.
            Think through the problem systematically before responding."""),
            ("user", "{context}\n\nQuery: {query}\n\nProvide your reasoning and answer:")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "query": query,
            "context": context or "No additional context provided."
        })
        
        return {
            "response": response.content,
            "reasoning_steps": "Step-by-step reasoning included in response",
            "agent": "text_reasoning"
        }

