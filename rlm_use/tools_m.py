import asyncio
import logging
import json
from abc import ABC, abstractmethod
from typing import Any

import chz
import google.genai as genai

#RLM Imports
from typing import Dict, List, Optional, Any 
from rlm.repl import REPLEnv

#Tinker Imports
from tinker_cookbook.renderers import Message, ToolCall

logger = logging.getLogger(__name__)


class ToolClientInterface(ABC):
    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def invoke(self, tool_call: ToolCall) -> list[Message]: ...

# May not need if I am passing in the environment from rlm_env
# @chz.chz
# class RLMToolClientConfig:
#     api_key: Optional[str] = None
#     recursive_model: str
#     max_iterations: int
#     depth: int = 0
#     context_data: dict | list | None
#     context_str: str | None

class RLMToolClient(ToolClientInterface):
    def __init__(
        self
    ):
        pass

#Next step: Replace the RLM schema with whats in 
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "execute_code",
                "title": "REPL Code Execution",
                "description": "Executes Python code in a REPL environment with access to context, llm_query function, and other REPL capabilities. Use this to analyze context, query sub-LLMs, and build up your answer.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in the REPL environment. The REPL has access to: context variable, llm_query() function, and standard Python built-ins. Code should be written as if it's being executed in a Python REPL."
                        }
                    },
                    "required": ["code"],
                },
                "outputSchema": {
                    "type": "string",
                    "description": "Execution results including stdout, stderr, and available variables"
                },
            }
        ]


    async def invoke(self, tool_call: ToolCall, rlm: REPLEnv) -> list[Message]:
        tool_name = tool_call.function.name
        if tool_name != "execute_code":
            raise ValueError(f"Invalid tool name: {tool_name}")
        
        try:
            args = tool_call.function.arguments
            args = json.loads(args)
            # print(type(args))
            # print("code" in args)
        except Exception as e:
            return [
                Message(role="tool", content=f"Error: {str(e)}")
            ]
        
        if not isinstance(args, dict) or "code" not in args:
            return [
                Message(role="tool", content="Error: 'code' parameter is required")
            ]
        
        code = args["code"]
        if not isinstance(code, str) or len(code.strip()) == 0:
            return [
                Message(role="tool", content="Error: 'code' must be a non-empty string")
            ]
        
        # Execute code using the existing REPL environment
        # This reuses your existing code execution infrastructure
        try:
            result = rlm.code_execution(code)
            
            logger.info(f"Code Executed, Result: {result}")
            # Format the result using your existing utility function
            from rlm.utils.utils import format_execution_result
            formatted_result = format_execution_result(
                result.stdout,
                result.stderr,
                result.locals,
            )
            
            # Build message content similar to how RLM_REPL does it
            message_content = f"Code executed:\n\n{code}\n```\n\nREPL output:\n{formatted_result}"
            
            return [Message(role="tool", content=message_content)]

        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            logger.error(error_msg)
            return [Message(role="tool", content=error_msg)]   