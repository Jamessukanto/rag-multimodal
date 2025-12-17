import logging
from typing import Optional
from contextlib import AsyncExitStack
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
import json
import os
from config import settings

# LLM provider imports
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = self._create_llm_client()
        self.tools = []
        self.messages = []
        self.logger = logging.getLogger(__name__)
    
    def _create_llm_client(self):
        """Create LLM client based on configuration"""
        provider = settings.llm_provider.lower()
        
        if provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package not installed")
            return Anthropic()
        elif provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed")
            return OpenAI()
        elif provider == "groq":
            if Groq is None:
                raise ImportError("groq package not installed")
            return Groq()
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    # connect to the MCP server
    async def connect_to_server(self, server_script_path: str):
        try:
            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python):
            # if not (is_python or is_js):
                raise ValueError("Server script must be a .py file for now")

            # Use uv run to ensure MCP server has its dependencies
            if is_python:
                server_dir = os.path.dirname(os.path.abspath(server_script_path))
                # Use uv --directory to run in the server's directory
                args = [
                    "--directory", server_dir,
                    "run", "python", os.path.basename(server_script_path)
                ]
                server_params = StdioServerParameters(
                    command="uv", args=args, env=None
                )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            self.logger.info("Connected to MCP server")

            mcp_tools = await self.get_mcp_tools()
            self.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in mcp_tools
            ]

            self.logger.info(
                f"Available tools: {[tool['function']['name'] for tool in self.tools]}"
            )
            self.logger.debug(f"Tools format: {self.tools[0] if self.tools else 'No tools'}")

            return True

        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise

    # get mcp tool list
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
            raise

    # process query
    async def process_query(self, query: str):
        try:
            self.logger.info(f"Processing query: {query}")
            if not self.tools:
                self.logger.warning("No tools available, reconnecting to server...")
                await self.connect_to_server(settings.server_script_path)
            user_message = {"role": "user", "content": query}
            self.messages = [user_message]

            provider = settings.llm_provider.lower()

            # ============================
            # Anthropic: full tool-calling
            # ============================
            if provider == "anthropic":
                while True:
                    response = await self.call_llm()

                    # Anthropic uses response.content (list of content blocks)
                    if response.content[0].type == "text" and len(response.content) == 1:
                        assistant_message = {
                            "role": "assistant",
                            "content": response.content[0].text,
                        }
                        self.messages.append(assistant_message)
                        await self.log_conversation()
                        break

                    # Tool call response
                    assistant_message = {
                        "role": "assistant",
                        "content": response.to_dict()["content"],
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()

                    for content in response.content:
                        if content.type == "tool_use":
                            tool_name = content.name
                            tool_args = content.input
                            tool_use_id = content.id
                            self.logger.info(
                                f"Calling tool {tool_name} with args {tool_args}"
                            )
                            try:
                                result = await self.session.call_tool(tool_name, tool_args)
                                self.logger.info(f"Tool {tool_name} result: {result}...")
                                self.messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_use_id,
                                                # Convert MCP result content (e.g. TextContent objects) to plain text
                                                "content": [
                                                    getattr(c, "text", str(c))
                                                    for c in (result.content or [])
                                                ],
                                            }
                                        ],
                                    }
                                )
                                await self.log_conversation()
                            except Exception as e:
                                self.logger.error(f"Error calling tool {tool_name}: {e}")
                                raise

                return self.messages

            # =============================
            # Groq (OpenAI-style) + tools
            # =============================
            if provider == "groq":
                # Internal messages for Groq (OpenAI chat format)
                groq_messages = [
                    {"role": "user", "content": query},
                ]
                # External messages for frontend (simple, consistent shape)
                external_messages = [
                    {"role": "user", "content": query},
                ]

                while True:
                    # Call Groq chat completions with tools
                    response = self.llm.chat.completions.create(
                        model=settings.llm_model,
                        max_tokens=settings.llm_max_tokens,
                        messages=groq_messages,
                        tools=self.tools if self.tools else None,
                    )

                    choice = response.choices[0]
                    message = choice.message

                    # Log what Groq returned for debugging
                    self.logger.debug(
                        f"Groq response - has tool_calls: {bool(getattr(message, 'tool_calls', None))}, "
                        f"content: {message.content[:100] if message.content else None}"
                    )

                    # If no tool calls → final answer
                    if not getattr(message, "tool_calls", None):
                        assistant_message = {
                            "role": "assistant",
                            "content": message.content,
                        }
                        external_messages.append(assistant_message)
                        self.messages = external_messages
                        await self.log_conversation()
                        break

                    # Tool calls present → record them for frontend,
                    # and prepare tool_call messages for Groq
                    # IMPORTANT: When tool_calls are present, content must be None (not empty string)
                    # This follows OpenAI/Groq tool calling protocol
                    tool_calls_content = []
                    assistant_msg_for_groq = {
                        "role": "assistant",
                        "content": None,  # Must be None when tool_calls are present
                        "tool_calls": [],
                    }

                    for tool_call in message.tool_calls:
                        # For frontend
                        tool_calls_content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments),
                            }
                        )

                        # For Groq conversation
                        assistant_msg_for_groq["tool_calls"].append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                    # Append assistant tool_call message for both views
                    external_messages.append(
                        {"role": "assistant", "content": tool_calls_content}
                    )
                    groq_messages.append(assistant_msg_for_groq)
                    await self.log_conversation()

                    # Now execute each tool via MCP and append tool results
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id

                        self.logger.info(
                            f"Calling MCP tool {tool_name} with args {tool_args}"
                        )
                        try:
                            result = await self.session.call_tool(tool_name, tool_args)
                            self.logger.info(f"Tool {tool_name} result: {result}...")

                            # Convert MCP result content (TextContent → text) for both
                            text_chunks = [
                                getattr(c, "text", str(c))
                                for c in (result.content or [])
                            ]
                            tool_text = "\n".join(text_chunks)

                            # Message for Groq (role=tool)
                            groq_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": tool_text,
                                }
                            )

                            # Message for frontend (tool_result)
                            external_messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_call_id,
                                            "content": text_chunks,
                                        }
                                    ],
                                }
                            )
                            self.messages = external_messages
                            await self.log_conversation()
                        except Exception as e:
                            self.logger.error(
                                f"Error calling MCP tool {tool_name}: {e}"
                            )
                            raise

                return external_messages

            # =============================
            # Fallback: provider not wired
            # =============================
            raise ValueError(
                f"LLM provider '{provider}' is not fully configured for tool use"
            )

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise

    # call llm
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM")
            provider = settings.llm_provider.lower()
            
            if provider == "anthropic":
                # Prepare API call parameters with tools for Anthropic
                api_params = {
                    "model": settings.llm_model,
                    "max_tokens": settings.llm_max_tokens,
                    "messages": self.messages,
                }
                if self.tools:
                    # Validate tools format
                    for i, tool in enumerate(self.tools):
                        if not isinstance(tool, dict):
                            raise ValueError(f"Tool {i} is not a dictionary: {tool}")
                        if "type" not in tool:
                            raise ValueError(f"Tool {i} missing 'type' field: {tool}")
                        if tool["type"] != "function":
                            raise ValueError(
                                f"Tool {i} has invalid type '{tool['type']}', expected 'function'"
                            )
                        if "function" not in tool:
                            raise ValueError(f"Tool {i} missing 'function' field: {tool}")
                    api_params["tools"] = self.tools

                return self.llm.messages.create(**api_params)
            elif provider == "openai":
                # OpenAI chat completions with tools (similar to Groq)
                api_params = {
                    "model": settings.llm_model,
                    "max_tokens": settings.llm_max_tokens,
                    "messages": self.messages,
                }
                if self.tools:
                    api_params["tools"] = self.tools
                return self.llm.chat.completions.create(**api_params)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            self.logger.error(f"Tools count: {len(self.tools) if self.tools else 0}")
            if self.tools:
                self.logger.error(f"First tool structure: {json.dumps(self.tools[0], indent=2)}")
                self.logger.error(f"All tools: {json.dumps(self.tools, indent=2)}")
            raise

    # cleanup
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected from MCP server")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            traceback.print_exc()
            raise

    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"], "content": []}

                # Handle both string and list content
                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(
                                content_item.to_dict()
                            )
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(
                                content_item.model_dump()
                            )
                        else:
                            serializable_message["content"].append(content_item)

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise