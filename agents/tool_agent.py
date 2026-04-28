"""Tool agent for executing external tools and actions."""
from typing import Dict, Any, List, Optional
from tools import WebSearchTool, PythonExecutor
from config import settings


class ToolAgent:
    """Handles tool execution with guardrails."""
    
    def __init__(self):
        """Initialize tool agent."""
        self.web_search: Optional[WebSearchTool] = WebSearchTool() if settings.enable_web_search else None
        self.python_exec: Optional[PythonExecutor] = PythonExecutor() if settings.enable_python_exec else None
    
    def execute(self, query: str) -> Dict[str, Any]:
        """
        Execute tools to answer the query.
        
        Args:
            query: User query requiring tool usage
            
        Returns:
            Dict with 'response', 'tools_used', 'tool_results'
        """
        tools_used: List[str] = []
        tool_results: List[Any] = []

        if not self.web_search and not self.python_exec:
            return {
                "response": "Инструменты отключены в настройках (ENABLE_WEB_SEARCH / ENABLE_PYTHON_EXEC).",
                "tools_used": tools_used,
                "tool_results": tool_results,
                "agent": "tool",
            }

        # Простая стратегия без LangChain-агента:
        # - если запрос явно похож на поиск, пробуем web_search
        # - если запрос выглядит как python-код (много '\n' или 'print(' и т.п.), пробуем python_executor
        lower = query.lower()
        try:
            if self.web_search and any(k in lower for k in ["search", "find", "lookup", "current", "latest", "news"]):
                tools_used.append("web_search")
                results = self.web_search.search(query, num_results=5)
                tool_results.append(results)
                if results:
                    formatted = "\n".join([f"- {r.get('title','')}: {r.get('url','')}" for r in results])
                    return {
                        "response": f"Результаты web search:\n{formatted}",
                        "tools_used": tools_used,
                        "tool_results": tool_results,
                        "agent": "tool",
                    }

            if self.python_exec and ("\n" in query or "print(" in query or "def " in query or "import " in query):
                tools_used.append("python_executor")
                result = self.python_exec.execute(query)
                tool_results.append(result)
                return {
                    "response": f"Результат Python:\n{result}",
                    "tools_used": tools_used,
                    "tool_results": tool_results,
                    "agent": "tool",
                }

            return {
                "response": "Не нашёл подходящего инструмента под запрос. Попробуй переформулировать (например, добавить 'search ...' или вставить Python-код).",
                "tools_used": tools_used,
                "tool_results": tool_results,
                "agent": "tool",
            }
        except Exception as e:
            return {
                "response": f"Ошибка при выполнении инструментов: {str(e)}",
                "tools_used": tools_used,
                "tool_results": tool_results,
                "agent": "tool",
                "error": str(e),
            }

