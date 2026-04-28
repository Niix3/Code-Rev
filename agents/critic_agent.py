"""Critic/Verifier agent for quality assurance."""
from typing import Dict, Any, List, Optional
from openai import OpenAI
from config import settings


class CriticAgent:
    """Verifies and critiques agent responses."""
    
    def __init__(self):
        """Initialize critic agent."""
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    
    def verify(self, query: str, response: str, agent_name: str, 
               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Verify and critique the response.
        
        Args:
            query: Original user query
            response: Agent response to verify
            agent_name: Name of the agent that generated the response
            context: Optional context from agent execution
            
        Returns:
            Dict with 'verified', 'score', 'critique', 'suggestions'
        """
        system_prompt = """You are a quality assurance agent. Evaluate the response for:
1. Relevance to the query
2. Accuracy and correctness
3. Completeness
4. Clarity and coherence

Provide a score from 0-10 and detailed feedback."""
        user_prompt = f"""Original Query: {query}
            
Agent: {agent_name}
Response: {response}

Context: {str(context) if context else "No additional context"}

Evaluate this response:"""

        critique_response = self.client.chat.completions.create(
            model=settings.default_llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        critique_text = critique_response.choices[0].message.content or ""
        
        # Extract score (simple heuristic)
        score = 7.0  # Default
        if "score" in critique_text.lower():
            try:
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)/10', critique_text)
                if score_match:
                    score = float(score_match.group(1))
            except:
                pass
        
        verified = score >= 7.0
        
        return {
            "verified": verified,
            "score": score,
            "critique": critique_text,
            "suggestions": critique_text,  # Could be parsed separately
            "agent": "critic"
        }
    
    def aggregate_responses(self, responses: List[Dict[str, Any]], 
                           query: str) -> Dict[str, Any]:
        """
        Aggregate multiple agent responses into final answer.
        
        Args:
            responses: List of agent responses
            query: Original query
            
        Returns:
            Aggregated response
        """
        if not responses:
            return {"response": "No responses to aggregate", "sources": []}
        
        # If only one response, return it
        if len(responses) == 1:
            return responses[0]
        
        # Combine responses
        combined = "\n\n".join([
            f"[{r.get('agent', 'unknown')}]: {r.get('response', '')}"
            for r in responses
        ])
        
        system_prompt = """You are synthesizing multiple agent responses into a coherent final answer.
Combine the best information from each response, resolve conflicts, and provide
a unified, comprehensive answer."""
        user_prompt = f"Query: {query}\n\nAgent Responses:\n{combined}\n\nSynthesized Answer:"
        final_response = self.client.chat.completions.create(
            model=settings.default_llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        # Collect all sources
        all_sources = []
        for r in responses:
            if "sources" in r:
                all_sources.extend(r["sources"])
        
        return {
            "response": final_response.choices[0].message.content or "",
            "sources": list(set(all_sources)),  # Remove duplicates
            "agent_responses": responses
        }

