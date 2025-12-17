# Multimodal RAG

Multimodal RAG system with MCP tool integration for dynamic retrieval.

## Quick Start

**Backend:**
```bash
cd api && uv sync && uv run python main.py
```
Runs on `http://localhost:8000` This starts MCP server automatically as it's set to stdio for transport

**Frontend:**
```bash
cd frontend && npm install && npm run dev
```
Runs on `http://localhost:3000`

## Setup

Create `.env`:
```
GROQ_API_KEY=your_key
SERPER_API_KEY=your_key
```

## Structure

- `api/` - FastAPI backend with MCP client
- `frontend/` - React chat UI
- `mcp_server_docs/` - MCP server (docs search tools)

Configure LLM in `api/config.py`.


## References

- mcp https://modelcontextprotocol.io/docs/develop/build-server#python
- enable mcp tool in claude desktop:
{
  "mcpServers": {
    "documentation": {
      "command": "/Users/jamessukanto/.local/bin/uv",
      "args": [
        "--directory",
        "/Users/jamessukanto/Desktop/codes/exps/mcp/mcp_server_docs",
        "run",
        "main.py"
      ]
    }
  }
}

