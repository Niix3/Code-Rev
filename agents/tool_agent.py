"""Tool agent for executing external tools and actions."""
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from tools import WebSearchTool, PythonExecutor
from config import settings


class ToolAgent:
    """Handles tool execution with guardrails."""
    
    def __init__(self):
        """Initialize tool agent."""
        self.llm = ChatOpenAI(
            model=settings.default_llm_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        
        # Initialize tools
        self.tools: List[Tool] = []
        
        # Add web search
        if settings.enable_web_search:
            web_search = WebSearchTool()
            self.tools.append(web_search.get_tool())
        
        # Add Python executor
        if settings.enable_python_exec:
            python_exec = PythonExecutor()
            self.tools.append(python_exec.get_tool())
        
        # Create agent
        self.agent_executor: Optional[AgentExecutor] = None
        if self.tools:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a tool-using agent. You have access to various tools.
                Use them wisely and only when necessary. Always explain what you're doing.
                You can make up to {max_tool_calls} tool calls per request."""),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}")
            ])
            
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=settings.max_tool_calls
            )
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute tools to answer the query.
        
        Args:
            query: User query requiring tool usage
            
        Returns:
            Dict with 'response', 'tools_used', 'tool_results'
        """
        if not self.agent_executor:
            return {
                "response": "No tools available. Tool agent is not properly configured.",
                "tools_used": [],
                "tool_results": [],
                "agent": "tool"
            }
        
        try:
            result = self.agent_executor.invoke({"input": query})
            
            return {
                "response": result.get("output", ""),
                "tools_used": [tool.name for tool in self.tools],
                "tool_results": result.get("intermediate_steps", []),
                "agent": "tool"
            }
        except Exception as e:
            return {
                "response": f"Error executing tools: {str(e)}",
                "tools_used": [],
                "tool_results": [],
                "agent": "tool",
                "error": str(e)
            }

