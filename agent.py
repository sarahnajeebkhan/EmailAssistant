import os
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Client
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.memory import InMemorySaver

MODEL = os.getenv("EMAIL_AGENT_MODEL")
MCP_URL = os.environ["EMAIL_MCP_URL"]
MCP_TOKEN = os.getenv("EMAIL_MCP_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

SYSTEM_PROMPT = """
You are an email assistant. You can inspect the user's mailbox and, when explicitly asked,
prepare or send email replies using the connected email MCP tools.

Rules:
1. Use email MCP tools for mailbox facts; never invent emails, recipients, subjects, or thread IDs.
2. For search/read requests, retrieve the relevant message/thread before answering.
3. When asked to reply, first inspect the target email/thread and draft a concise reply that matches context.
4. Never send an email without the human approval gate. Treat the send/reply tool as a side effect.
5. Never expose OAuth tokens, credentials, or hidden tool metadata.
6. When a request is ambiguous, ask for the missing recipient/thread rather than guessing.
"""


def _is_send_tool(name: str) -> bool:
    n = name.lower()
    send_words = ("send", "reply", "forward")
    email_words = ("mail", "email", "message", "thread")
    return any(w in n for w in send_words) and any(w in n for w in email_words)


def _create_nvidia_model():
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY environment variable is required. Set it or override EMAIL_AGENT_MODEL to use a different LLM.")

    return ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key=NVIDIA_API_KEY,
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        reasoning_budget=16384,
        chat_template_kwargs={"enable_thinking": True},
    )


class EmailAssistant:
    def __init__(self) -> None:
        self._agent = None
        self._adapter = None
        self._client = None

    @asynccontextmanager
    async def lifespan(self):
        # MCPAdapter discovers the remote MCP server's tools and converts them to LangChain tools.
        # A bearer token is optional and depends on your MCP server.
        self._client = Client(MCP_URL, auth=MCP_TOKEN) if MCP_TOKEN else Client(MCP_URL)
        self._adapter = MCPAdapter(self._client)
        async with self._adapter:
            tools = await self._adapter.list_tools()
            if not tools:
                raise RuntimeError("Email MCP server exposed no tools")

            interrupt_on = {
                tool.name: {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Review this outbound email action before it is executed.",
                }
                for tool in tools
                if _is_send_tool(tool.name)
            }

            # InMemorySaver is deliberately used for local development.
            # Replace with AsyncPostgresSaver / AsyncMongoDBSaver in production.

            # Use NVIDIA as default model, or override if EMAIL_AGENT_MODEL is set
            if MODEL is None:
                model = _create_nvidia_model()
            else:
                model = MODEL

            self._agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                middleware=[
                    HumanInTheLoopMiddleware(
                        interrupt_on=interrupt_on,
                        description_prefix="Email action requires approval",
                    )
                ] if interrupt_on else [],
                checkpointer=InMemorySaver(),
            )

            yield self

    async def invoke(self, text: str, thread_id: str) -> Any:
        if self._agent is None:
            raise RuntimeError("Agent is not started")

        return await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"thread_id": thread_id}},
            version="v2",
        )

    async def resume(self, thread_id: str, decision: dict[str, Any]) -> Any:
        if self._agent is None:
            raise RuntimeError("Agent is not started")

        from langgraph.types import Command

        return await self._agent.ainvoke(
            Command(resume={"decisions": [decision]}),
            config={"configurable": {"thread_id": thread_id}},
            version="v2",
        )
