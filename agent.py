from contextlib import asynccontextmanager
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_google_community import GmailToolkit, CalendarToolkit
from langchain_google_community.search import GoogleSearchAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver

from config import settings

SYSTEM_PROMPT = """
You are a productivity assistant with access to Google services including Gmail, Calendar, and Search.
You can inspect the user's mailbox, manage calendar events, and perform web searches.

Rules:
1. Use Google tools for factual information; never invent emails, recipients, subjects, or calendar events.
2. For search/read requests, retrieve the relevant data before answering.
3. When asked to reply to emails, first inspect the target email/thread and draft a concise reply that matches context.
4. Never send an email without the human approval gate. Treat the send/reply tool as a side effect.
5. Never expose OAuth tokens, credentials, or hidden tool metadata.
6. When a request is ambiguous, ask for the missing information rather than guessing.
7. For calendar operations, verify existing events before creating new ones to avoid conflicts.
"""


def _is_send_tool(name: str) -> bool:
    n = name.lower()
    send_words = ("send", "reply", "forward")
    email_words = ("mail", "email", "message", "thread")
    return any(w in n for w in send_words) and any(w in n for w in email_words)


class EmailAssistant:
    def __init__(self) -> None:
        self._agent = None
        self._tools = None

    @asynccontextmanager
    async def lifespan(self):
        # Initialize Google toolkits
        gmail_toolkit = GmailToolkit()
        calendar_toolkit = CalendarToolkit()

        # Collect all tools from the toolkits
        tools = []
        tools.extend(gmail_toolkit.get_tools())
        tools.extend(calendar_toolkit.get_tools())

        # Add Google Search if API key and CSE ID are configured
        if settings.google_api_key and settings.google_cse_id:
            from langchain.tools import StructuredTool
            from pydantic import BaseModel, Field

            class GoogleSearchInput(BaseModel):
                query: str = Field(description="Search query for Google")

            search_wrapper = GoogleSearchAPIWrapper(
                google_api_key=settings.google_api_key,
                google_cse_id=settings.google_cse_id,
                k=5
            )

            google_search_tool = StructuredTool.from_function(
                func=search_wrapper.run,
                name="google_search",
                description="Search Google for recent results",
                args_schema=GoogleSearchInput
            )
            tools.append(google_search_tool)

        if not tools:
            raise RuntimeError("No Google tools available")

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
