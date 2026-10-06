from contextlib import asynccontextmanager
from typing import Any

from fastmcp import Client
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.mcp import MCPAdapter
from langgraph.checkpoint.memory import InMemorySaver

from config import settings

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


class EmailAssistant:
    def __init__(self) -> None:
        self._agent = None
        self._adapter = None
        self._client = None

    @asynccontextmanager
    async def lifespan(self):
        # MCPAdapter discovers the remote MCP server's tools and converts them to LangChain tools.
        # A bearer token is optional and depends on your MCP server.
        self._client = (
            Client(settings.email_mcp_url, auth=settings.email_mcp_token)
            if settings.email_mcp_token
            else Client(settings.email_mcp_url)
        )
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

            model = settings.get_model()

            self._agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=SYSTEM_PROMPT,
                middleware=[
                    HumanInTheLoopMiddleware(
                        interrupt_on=interrupt_on,
                        description_prefix="Email action requires approval",
                    )
                ]
                if interrupt_on
                else [],
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
