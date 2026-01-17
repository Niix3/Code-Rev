"""Critic/Verifier agent for quality assurance."""
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import settings


class CriticAgent:
    """Verifies and critiques agent responses."""
    
    def __init__(self):
        """Initialize critic agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=0.1,  # Lower temperature for critical analysis
            api_key=settings.openai_api_key
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
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance agent. Evaluate the response for:
            1. Relevance to the query
            2. Accuracy and correctness
            3. Completeness
            4. Clarity and coherence
            
            Provide a score from 0-10 and detailed feedback."""),
            ("user", """Original Query: {query}
            
Agent: {agent_name}
Response: {response}

Context: {context}

Evaluate this response:""")
        ])
        
        chain = prompt | self.llm
        critique_response = chain.invoke({
            "query": query,
            "agent_name": agent_name,
            "response": response,
            "context": str(context) if context else "No additional context"
        })
        
        # Parse critique (simplified - in production use structured output)
        critique_text = critique_response.content
        
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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are synthesizing multiple agent responses into a coherent final answer.
            Combine the best information from each response, resolve conflicts, and provide
            a unified, comprehensive answer."""),
            ("user", "Query: {query}\n\nAgent Responses:\n{responses}\n\nSynthesized Answer:")
        ])
        
        chain = prompt | self.llm
        final_response = chain.invoke({
            "query": query,
            "responses": combined
        })
        
        # Collect all sources
        all_sources = []
        for r in responses:
            if "sources" in r:
                all_sources.extend(r["sources"])
        
        return {
            "response": final_response.content,
            "sources": list(set(all_sources)),  # Remove duplicates
            "agent_responses": responses
        }

