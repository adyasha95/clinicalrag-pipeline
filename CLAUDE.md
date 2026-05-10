# ClinicalRAG

A retrieval-augmented generation system for querying clinical trial records from ClinicalTrials.gov.

## Tech Stack

- **Language**: Python 3.11
- **API framework**: FastAPI (async throughout)
- **LLM orchestration**: LangChain
- **Vector store**: Pinecone
- **LLM provider**: Anthropic API (Claude)
- **Config**: python-dotenv + `.env`
- **Containerization**: Docker
- **Deployment**: Google Cloud Run

## Project Layout

```
clinicalrag/
├── src/                  # All application source code
│   ├── api/              # FastAPI routers and request/response models
│   ├── ingestion/        # ClinicalTrials.gov data fetch and chunking
│   ├── retrieval/        # Pinecone indexing and similarity search
│   ├── generation/       # LangChain chains and Anthropic API calls
│   └── core/             # Shared config, logging, dependencies
├── tests/                # Pytest test suite (mirrors src/ structure)
├── data/                 # Raw and processed clinical trial data files
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example          # Committed template — never commit .env
└── CLAUDE.md
```

## Development Rules

### Python Version
Always use **Python 3.11**. Do not use syntax or stdlib features introduced after 3.11.

### FastAPI: Always Async
All FastAPI route handlers, dependencies, and service layer functions must be `async def`. Use `httpx.AsyncClient` for outbound HTTP (e.g., ClinicalTrials.gov API calls), not `requests`. Use async-compatible Pinecone and LangChain interfaces where available.

```python
# correct
@router.get("/trials/{nct_id}")
async def get_trial(nct_id: str) -> TrialResponse:
    ...

# wrong — never do this in a route handler
@router.get("/trials/{nct_id}")
def get_trial(nct_id: str) -> TrialResponse:
    ...
```

### API Keys and Secrets
**Never hardcode API keys, tokens, or credentials anywhere in source code or tests.**

- All secrets live in `.env` (gitignored)
- `.env.example` is committed with placeholder values documenting required variables
- Load config via `python-dotenv` in a single `src/core/config.py` using pydantic `BaseSettings`
- Required env vars (document in `.env.example`):
  - `ANTHROPIC_API_KEY`
  - `PINECONE_API_KEY`
  - `PINECONE_INDEX_NAME`
  - `PINECONE_ENVIRONMENT`
  - `CLINICALTRIALS_API_BASE_URL` (default: `https://clinicaltrials.gov/api/v2`)

### Dependency Management
Use `pyproject.toml` with a lockfile. Do not use `requirements.txt` as the primary dependency file.

## Common Commands

```bash
# Install dependencies (assumes uv or pip with pyproject.toml)
uv sync

# Run the dev server
uvicorn src.main:app --reload --port 8080

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Build Docker image
docker build -t clinicalrag .

# Run with Docker Compose
docker-compose up
```

## Testing

- Tests live in `tests/`, mirroring `src/` structure (e.g., `tests/retrieval/` for `src/retrieval/`)
- Use `pytest` with `pytest-asyncio` for async test functions
- Mark async tests with `@pytest.mark.asyncio`
- Mock external services (Pinecone, Anthropic, ClinicalTrials.gov) in unit tests — do not make live API calls in tests
- Integration tests that require live services should be in `tests/integration/` and skipped by default (`pytest -m "not integration"`)

## Architecture Notes

- **Ingestion pipeline**: fetches trial records from ClinicalTrials.gov, chunks them, embeds via Anthropic or a compatible embedder, and upserts into Pinecone
- **Query pipeline**: embeds user query → Pinecone similarity search → retrieved chunks passed to LangChain chain → Claude generates grounded answer
- **API layer**: thin FastAPI wrapper over the query pipeline; ingestion triggered via admin endpoints or background tasks
