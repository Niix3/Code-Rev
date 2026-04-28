"""Text reasoning agent for complex reasoning tasks."""
from typing import Dict, Any, Optional
from openai import OpenAI
from config import settings


class TextReasoningAgent:
    """Handles complex reasoning and analysis tasks."""
    
    def __init__(self):
        """Initialize text reasoning agent."""
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
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
        system_prompt = """You are an expert reasoning agent. Analyze the problem step by step,
show your reasoning process, and provide a clear, well-reasoned answer.
Think through the problem systematically before responding."""
        user_prompt = (
            f"{context or 'No additional context provided.'}\n\n"
            f"Query: {query}\n\n"
            "Provide your reasoning and answer:"
        )
        completion = self.client.chat.completions.create(
            model=settings.default_llm_model,
            temperature=settings.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_text = completion.choices[0].message.content or ""
        
        return {
            "response": response_text,
            "reasoning_steps": "Step-by-step reasoning included in response",
            "agent": "text_reasoning"
        }

