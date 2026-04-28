"""LangGraph orchestrator for multi-agent system."""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from agents import (
    RouterAgent,
    TextReasoningAgent,
    ToolAgent,
    CriticAgent
)


class AgentState(TypedDict):
    """State shared across all agents."""
    query: str
    routed_agent: str
    agent_response: dict
    all_responses: Annotated[list, operator.add]
    final_response: str
    verified: bool
    critic_verification: dict
    iteration: int


class LangGraphOrchestrator:
    """Main orchestrator using LangGraph."""
    
    def __init__(self):
        """Initialize orchestrator with all agents."""
        self.router = RouterAgent()
        self.text_agent = TextReasoningAgent()
        self.tool_agent = ToolAgent()
        self.critic = CriticAgent()
        
        # Build graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("text_reasoning", self._text_reasoning_node)
        workflow.add_node("tool", self._tool_node)
        workflow.add_node("critic", self._critic_node)
        workflow.add_node("aggregate", self._aggregate_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "text_reasoning": "text_reasoning",
                "tool": "tool"
            }
        )
        
        # All agents go to critic
        workflow.add_edge("text_reasoning", "critic")
        workflow.add_edge("tool", "critic")
        
        # Critic goes to aggregate
        workflow.add_edge("critic", "aggregate")
        
        # Aggregate is the end
        workflow.add_edge("aggregate", END)
        
        return workflow
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router node - determines which agent to use."""
        routing = self.router.route(
            state["query"]
        )
        return {
            "routed_agent": routing["agent"]
        }
    
    def _text_reasoning_node(self, state: AgentState) -> AgentState:
        """Text reasoning agent node."""
        context = state.get("agent_response", {}).get("response", "")
        response = self.text_agent.reason(state["query"], context)
        return {
            "agent_response": response,
            "all_responses": [response]
        }
    
    def _tool_node(self, state: AgentState) -> AgentState:
        """Tool agent node."""
        response = self.tool_agent.execute(state["query"])
        return {
            "agent_response": response,
            "all_responses": [response]
        }
    
    def _critic_node(self, state: AgentState) -> AgentState:
        """Critic/verifier node."""
        agent_response = state.get("agent_response", {})
        verification = self.critic.verify(
            state["query"],
            agent_response.get("response", ""),
            agent_response.get("agent", "unknown"),
            agent_response
        )
        return {
            "verified": verification["verified"],
            "critic_verification": verification,
            "agent_response": {**agent_response, "verification": verification}
        }
    
    def _aggregate_node(self, state: AgentState) -> AgentState:
        """Response aggregation node."""
        all_responses = state.get("all_responses", [])
        if not all_responses:
            all_responses = [state.get("agent_response", {})]
        
        aggregated = self.critic.aggregate_responses(all_responses, state["query"])
        verification = state.get("critic_verification", {})
        if verification and "verification" not in aggregated:
            aggregated = {**aggregated, "verification": verification}
        return {
            "final_response": aggregated.get("response", ""),
            "agent_response": aggregated
        }
    
    def _route_decision(self, state: AgentState) -> str:
        """Decision function for routing."""
        return state.get("routed_agent", "text_reasoning")
    
    def invoke(self, query: str) -> dict:
        """
        Process a request through the orchestrator.
        
        Args:
            query: User query
            
        Returns:
            Final response dict
        """
        initial_state = {
            "query": query,
            "routed_agent": "",
            "agent_response": {},
            "all_responses": [],
            "final_response": "",
            "verified": False,
            "critic_verification": {},
            "iteration": 0
        }
        
        result = self.app.invoke(initial_state)
        return result

