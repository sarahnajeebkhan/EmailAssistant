from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agent import EmailAssistant

assistant = EmailAssistant()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with assistant.lifespan():
        yield


app = FastAPI(title="Email Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class DecisionRequest(BaseModel):
    thread_id: str
    type: str
    message: str | None = None
    edited_action: dict | None = None


def serialize_result(result):
    out = {"status": "completed", "interrupts": []}
    if hasattr(result, "interrupts") and result.interrupts:
        out["status"] = "needs_approval"
        out["interrupts"] = [i.value for i in result.interrupts]
    value = getattr(result, "value", result)
    messages = value.get("messages", []) if isinstance(value, dict) else []
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is not None:
            out["answer"] = content
    return out


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = await assistant.invoke(req.message, req.thread_id)
        return serialize_result(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/approve")
async def approve(req: DecisionRequest):
    decision = {"type": req.type}
    if req.message:
        decision["message"] = req.message
    if req.edited_action:
        decision["edited_action"] = req.edited_action
    try:
        result = await assistant.resume(req.thread_id, decision)
        return serialize_result(result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
