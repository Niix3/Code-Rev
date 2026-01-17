"""LangGraph orchestrator for multi-agent system."""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from agents import (
    RouterAgent,
    TextReasoningAgent,
    VisionAgent,
    RetrievalAgent,
    ToolAgent,
    CriticAgent
)
from rag import VectorStore


class AgentState(TypedDict):
    """State shared across all agents."""
    query: str
    image: Annotated[list, operator.add]  # For image data
    has_image: bool
    routed_agent: str
    agent_response: dict
    all_responses: Annotated[list, operator.add]
    final_response: str
    verified: bool
    iteration: int


class LangGraphOrchestrator:
    """Main orchestrator using LangGraph."""
    
    def __init__(self, vector_store: VectorStore = None):
        """Initialize orchestrator with all agents."""
        self.router = RouterAgent()
        self.text_agent = TextReasoningAgent()
        self.vision_agent = VisionAgent()
        self.retrieval_agent = RetrievalAgent(vector_store)
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
        workflow.add_node("vision", self._vision_node)
        workflow.add_node("retrieval", self._retrieval_node)
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
                "vision": "vision",
                "retrieval": "retrieval",
                "tool": "tool"
            }
        )
        
        # All agents go to critic
        workflow.add_edge("text_reasoning", "critic")
        workflow.add_edge("vision", "critic")
        workflow.add_edge("retrieval", "critic")
        workflow.add_edge("tool", "critic")
        
        # Critic goes to aggregate
        workflow.add_edge("critic", "aggregate")
        
        # Aggregate is the end
        workflow.add_edge("aggregate", END)
        
        return workflow
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router node - determines which agent to use."""
        routing = self.router.route(
            state["query"],
            has_image=state.get("has_image", False)
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
    
    def _vision_node(self, state: AgentState) -> AgentState:
        """Vision agent node."""
        image = state.get("image", [None])[0] if state.get("image") else None
        if not image:
            # Fallback to text reasoning if no image
            response = self.text_agent.reason(state["query"])
        else:
            response = self.vision_agent.analyze(state["query"], image)
        return {
            "agent_response": response,
            "all_responses": [response]
        }
    
    def _retrieval_node(self, state: AgentState) -> AgentState:
        """Retrieval agent node."""
        response = self.retrieval_agent.retrieve(state["query"])
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
            "agent_response": {**agent_response, "verification": verification}
        }
    
    def _aggregate_node(self, state: AgentState) -> AgentState:
        """Response aggregation node."""
        all_responses = state.get("all_responses", [])
        if not all_responses:
            all_responses = [state.get("agent_response", {})]
        
        aggregated = self.critic.aggregate_responses(all_responses, state["query"])
        return {
            "final_response": aggregated.get("response", ""),
            "agent_response": aggregated
        }
    
    def _route_decision(self, state: AgentState) -> str:
        """Decision function for routing."""
        return state.get("routed_agent", "text_reasoning")
    
    def invoke(self, query: str, image=None) -> dict:
        """
        Process a request through the orchestrator.
        
        Args:
            query: User query
            image: Optional image (PIL Image, bytes, or file path)
            
        Returns:
            Final response dict
        """
        initial_state = {
            "query": query,
            "image": [image] if image else [],
            "has_image": image is not None,
            "routed_agent": "",
            "agent_response": {},
            "all_responses": [],
            "final_response": "",
            "verified": False,
            "iteration": 0
        }
        
        result = self.app.invoke(initial_state)
        return result

