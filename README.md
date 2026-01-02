# Multimodal RAG with agentic tool calling

Retrieves relevant information from ingested documents using **multimodal embedding-based semantic search** (PDFs with text and images) and tools via **Model Context Protocol (MCP)**. Two-stage retrieval: ANN search with single-vector embeddings, then reranking with multi-vector embeddings. 

<details>
<summary>Agentic Tool System (MCP + Internal Services Integration)</summary>

The system uses a unified tool registry that combines:

1. **External MCP Tools**: Tools from external MCP servers (stdio communication) See mcp_server_docs/
   - Wrapped via `MCPToolAdapter` to match internal tool interface
   - Examples: web search, URL fetching (from `mcp_server_docs/`)

2. **Internal Internal Services Tools**: Direct function calls within the backend
   - Implement `BaseTool` interface directly
   - Example: `retrieve_documents` (semantic document search)

Both tool types are registered in `ToolRegistry` and appear identical to the LLM via `AgentOrchestrator`.

</details>


### Directory

```
.
├── backend/                      
│   ├── api/                          
│   │   ├── routes/                   
│   │   └── schemas/                  
│   ├── core/                         
│   │   ├── config.py                 
│   │   ├── exceptions.py             
│   │   └── startup.py                
│   ├── domain/                       # Business logic
│   │   ├── agentic/                  # Agent orchestration, mcp client, tools
│   │   ├── rag/                      # Chunking, embedding, retrieval
│   │   └── evaluation/               # Retrieval evaluation
│   ├── services/                     # Orchestrates domain logic
│   ├── storage/                      
│   │   ├── document_sql_store.py     # PostgreSQL (documents & chunks metadata)
│   │   ├── single_vector_store.py    # ChromaDB (single-vector embeddings)
│   │   ├── multi_vector_store.py     # Mmap numpy (multi-vector embeddings)
│   │   └── file_store.py             # Filesystem
│   └── data/                        
│       ├── documents/                # Uploaded PDFs
│       ├── chunks/                   # Uploaded PDFs chunked by page
│       ├── single_vector_db/         
│       └── multi_vector_db/          
├── frontend/                        # React frontend (Vite)
├── mcp_server_docs/                 # External MCP server (stdio)
├── docker-compose.yml               # Currently just Postgres Service
└── Makefile                        
```

## Quick Start

**Backend:**
```bash
# Backend
# Runs on `http://localhost:8000` 
# Auto-starts MCP server as it's set to stdio for transport
make dev-back

# Frontend - WIP
# Runs on `http://localhost:3000`
make dev-front

```

## APIs

Once the backend is running, API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs` 
- **ReDoc**: `http://localhost:8000/redoc` 











<details>
<summary>Tech Stack</summary>

### Backend

| Area | Component | Immediate (Dev / Early Prod) | Later  |
|---|---|---|---|
| Web | Framework | FastAPI | — |
|  | ASGI Server | Uvicorn | Gunicorn |
| MCP | MCP Server | Local (stdio) | — |
| Data | Database | Postgres (Docker) | Supabase / Render |
|  | Filestore | Local | S3 / GCS |
|  | Single-vector store | ChromaDB (local) | ChromaDB Cloud |
|  | Multiple-vector store | mmap numpy files | — |
|  | ORM | SQLAlchemy | — |
|  | Migrations | Alembic | — |
| API Boundary | Validation | Pydantic  | — |
|  | Versioning | URL-based  | Backward-compatible schema evolution |
| Auth & Identity | Identity Provider | Backend-issued JWT | Clerk |
|  | Auth Flow | JWT verification in FastAPI | OAuth handled by IDP |
| Config & Secrets | Settings Loader | pydantic-settings | — |
|  | Secrets Source | `.env` files | Platform env vars + secret manager |
| Testing | Tests | pytest (CI-enforced) | — |
|  | Linting | black, ruff (CI-enforced) | — |
| Deployment | Containers | Docker Compose | Managed or Kubernetes |
|  | CI/CD | Critical only | Automated pipeline |
|  | Health Checks | — | `/health`, DB checks, worker heartbeat |
| Observability | Logging | — | Structured logging (structlog) |
|  | Metrics / Tracing | — | Prometheus / Grafana |
| Background Jobs | Job Queue | — | Celery |
|  | Broker | — | Redis |
|  | Scheduler | — | Celery Beat |
| Caching | Cache Layer | — | Redis |

<br>

### Frontend

| Area | Component | Immediate (MVP / Early) | Later (Scale / Mature) |
|---|---|---|---|
| Application | Framework | Next.js | — |
| Rendering | Strategy | Client-side (CSR) | SSR / SSG |
| Styling | CSS | Tailwind CSS | shadcn/ui, Radix |
| Data Fetching | Server State | fetch | TanStack Query |
| Client State | UI State | React state | Zustand |
| Auth | Identity UI | Clerk SDK | — |
| Forms | Form State | react-hook-form | react-hook-form + Zod |
| Validation | Schema | Inline validation | Zod schemas |
| Tooling | Build System | Next.js defaults | — |
| Testing | Unit / E2E | — | Jest / Playwright |
| Deployment | Hosting | Vercel | CDN / Edge |

</details>




<details>
<summary>Query Flow</summary>

As user sends a question...

| Step 1 | Step 2 | Step 3 |
|------|-----------|--------|
| **Unicorn Server:** Receives HTTP POST request from frontend and passes parsed request to FastAPI | **FastAPI:** Matches `@router.post("/query")` → Routes to `handle_mcp_query` → Calls dependencies and validates against `QueryRequest` model | **MCP Client:** Calls async `mcp_client.process_query` → LLM + MCP Tools (external MCP server and internal tools) → Response |


</details>

<details>
<summary>Inspect Data (DEV)</summary>

### PostgreSQL Inspection

```bash
brew install --cask dbeaver-community      
# If permission errors, fix permissions once, then retry 
sudo chown -R $(whoami) /usr/local/var/homebrew
```

---

### Python REPL for Single-Vector and Multi-Vector stores

```bash
cd backend && uv run python

import chromadb
from pathlib import Path
import pickle
import numpy as np
import json
from pprint import pprint

# ---------------- ChromaDB ----------------

client = chromadb.PersistentClient(
    path=str(Path("data/single_vector_db"))
)

collection = client.get_collection("embeddings")
print(f"\nCollection: {collection.name}. Total vectors: {collection.count()}\n")

# Get sample data (must explicitly include embeddings)
sample = collection.get(
    limit=1,
    include=['embeddings', 'metadatas', 'documents']
    # Note: ChromaDB's get() doesn't return embeddings/metadatas/documents by default
    # (for performance - embeddings can be large). Must explicitly include them.
)

# Investigate metadata

print(f"\nSample ID: {sample.get('ids')[0]}\n")
print("Metadata:")
pprint(sample.get('metadatas', []), indent=4, width=100)

# Investigate embedding
# Embedding first 5 dimensions

embeddings = sample.get('embeddings')
print(f"  Shape: {len(embeddings[0])} dimensions")
print(f"  First 5 values: {embeddings[0][:5]}")

# Query random vector
query_vec = embeddings[0]  
results = collection.query(
    query_embeddings=[query_vec],  
    n_results=2,
    include=['embeddings', 'metadatas', 'documents', 'distances']  
)

# Query results
pprint(results, indent=2, width=100)


# ---------------- mmap files ----------------

index_path = Path("data/multi_vector_db/multi_vector_index.pkl")
with open(index_path, "rb") as f:
    index = pickle.load(f)

print(f"\nTotal chunks: {len(index)}\n")
print("\nFirst 2 chunk IDs:", list(index.keys())[:2])
print()

# First chunk details

first_id = list(index.keys())[0]
emb = index[first_id]
print(f"\nChunk {first_id}:")

vec = np.array(emb)
print(f"\nVector shape: {vec.shape}, dtype: {vec.dtype}")

```

</details>


<details>
<summary>References</summary>

- mcp https://modelcontextprotocol.io/docs/develop/build-server#python
- enable mcp tool in claude desktop:

```bash
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
```
</details>