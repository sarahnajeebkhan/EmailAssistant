# Email Assistant (LangChain + MCP)

## What it does

- Reads/searches email through an MCP server.
- Uses the email thread as context before drafting a reply.
- Gates outbound send/reply/forward tool calls behind human approval.
- Maintains conversation state by `thread_id`.

## Supported Models

The assistant uses NVIDIA Nemotron 3.5 Lightning with reasoning by default. You can override this by setting the `EMAIL_AGENT_MODEL` environment variable:

- **NVIDIA** (default): Nemotron 3.5 Lightning with reasoning enabled
- **Other models**: Set `EMAIL_AGENT_MODEL` to your preferred model (e.g., `openai:gpt-5.5`)

## Run

```bash
python -m venv .venv
# source .venv/bin/activate
.venv\Scripts\activate.bat
pip install -r requirements.txt
cp .env.example .env
export EMAIL_MCP_URL=https://your-email-mcp.example.com/mcp
export EMAIL_MCP_TOKEN=...
export NVIDIA_API_KEY=your_nvidia_api_key
uvicorn api:app --reload
```

## Example requests

Search mailbox:
```json
POST /chat
{"thread_id":"chat-123","message":"Find the latest email from John about the Q4 review and summarize it."}
```

Draft and send after approval:
```json
POST /chat
{"thread_id":"chat-123","message":"Reply to that thread saying I can meet Thursday after 3 PM."}
```

When the response is `needs_approval`, show the proposed tool call to the user and then call:
```json
POST /approve
{"thread_id":"chat-123","type":"approve"}
```

To edit before sending, use `type=edit` with the `edited_action` returned by the interrupt.
