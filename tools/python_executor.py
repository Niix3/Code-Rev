"""Python code execution tool with safety guardrails."""
from typing import Dict, Any, Optional, Tuple
import subprocess
import sys
import io
import contextlib
from langchain_core.tools import Tool
from config import settings


class PythonExecutor:
    """Safe Python code execution with guardrails."""
    
    def __init__(self):
        """Initialize Python executor."""
        self.enabled = settings.enable_python_exec
        self.allowed_modules = {
            "math", "numpy", "pandas", "json", "datetime", 
            "collections", "itertools", "re", "string"
        }
        self.blocked_keywords = {
            "import os", "import sys", "import subprocess",
            "__import__", "eval", "exec", "open(", "file(",
            "input(", "raw_input(", "compile("
        }
    
    def _check_safety(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if code is safe to execute.
        
        Returns:
            (is_safe, error_message)
        """
        code_lower = code.lower()
        
        # Check for blocked keywords
        for keyword in self.blocked_keywords:
            if keyword.lower() in code_lower:
                return False, f"Blocked keyword detected: {keyword}"
        
        # Check for dangerous imports
        lines = code.split("\n")
        for line in lines:
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                module = line.split()[1].split(".")[0]
                if module not in self.allowed_modules:
                    return False, f"Import of '{module}' is not allowed"
        
        return True, None
    
    def execute(self, code: str, timeout: int = 10) -> Dict[str, Any]:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
            
        Returns:
            Dict with 'output', 'error', 'success' keys
        """
        if not self.enabled:
            return {
                "success": False,
                "error": "Python execution is disabled",
                "output": ""
            }
        
        # Safety check
        is_safe, error_msg = self._check_safety(code)
        if not is_safe:
            return {
                "success": False,
                "error": error_msg,
                "output": ""
            }
        
        # Execute in isolated environment
        try:
            # Capture stdout and stderr
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()
            
            with contextlib.redirect_stdout(output_buffer):
                with contextlib.redirect_stderr(error_buffer):
                    # Execute in subprocess with timeout
                    result = subprocess.run(
                        [sys.executable, "-c", code],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
            
            output = output_buffer.getvalue() or result.stdout
            error = error_buffer.getvalue() or result.stderr
            
            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Execution timeout after {timeout} seconds",
                "output": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    def get_tool(self) -> Tool:
        """Get LangChain Tool wrapper."""
        return Tool(
            name="python_executor",
            description="Execute Python code safely. Input should be valid Python code. Only basic libraries are allowed.",
            func=lambda code: str(self.execute(code))
        )

