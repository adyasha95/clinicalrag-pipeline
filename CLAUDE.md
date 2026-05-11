# ClinicalRAG

ClinicalRAG is an end-to-end retrieval-augmented generation pipeline for querying clinical trial records from ClinicalTrials.gov. It fetches and enriches trial data using the Anthropic Claude API, embeds records into a Pinecone vector database, and exposes a conversational chat interface via a FastAPI backend deployed on Google Cloud Run. Users can ask natural-language questions about trials and receive grounded answers with citations to NCT IDs.

---

## Tech Stack

| Layer | Tool / Service | Purpose |
|---|---|---|
| Data ingestion | ClinicalTrials.gov v2 API, `curl_cffi`, `tenacity` | Fetch paginated trial records; bypass TLS fingerprinting with Chrome impersonation |
| Enrichment | Anthropic Claude API (`claude-sonnet-4-20250514`) | Parse eligibility criteria into structured JSON; extract age ranges, complexity scores, condition tags |
| Embedding | `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dims) | Local, free embeddings — no OpenAI key required |
| Vector DB | Pinecone serverless (AWS `us-east-1`, cosine similarity) | Store and query trial embeddings with metadata filtering |
| Orchestration | LangChain 1.x (LCEL chains + LangGraph agent) | `build_qa_chain` for simple queries; `build_agent` for filter-aware tool-calling |
| LLM | Anthropic `claude-sonnet-4-20250514` | Generate grounded answers; cited by NCT ID |
| API | FastAPI (async, Pydantic v2, CORS) | `POST /query`, `POST /chat`, `GET /health` |
| UI | Vanilla HTML/CSS/JS (served via FastAPI `StaticFiles`) | Chat interface with multi-turn support, source chips, phase/condition filters |
| Observability | LangSmith | End-to-end chain tracing, token usage, latency |
| Containerization | Docker multi-stage build (`python:3.11-slim`) | Lean runtime image; `.env` never copied in |
| Deployment | Google Cloud Run (`europe-west3`, min=0, max=3, 1Gi) | Serverless, scales to zero when idle |
| CI/CD | GitHub Actions → Google Artifact Registry → Cloud Run | pytest → docker build/push → gcloud run deploy on every push to `main` |

---

## Project Structure

```
clinicalrag/
├── src/
│   ├── ingestion/
│   │   └── fetch_trials.py          # Fetch + paginate ClinicalTrials.gov; writes data/raw/trials.jsonl
│   ├── enrichment/
│   │   └── enrich_trial.py          # Claude API enrichment; writes data/enriched/trials_enriched.jsonl
│   ├── embedding/
│   │   └── embed_and_upsert.py      # fastembed + Pinecone upsert
│   ├── retrieval/                   # Vector search utilities (placeholder for direct retrieval helpers)
│   ├── agent/
│   │   └── chain.py                 # build_qa_chain() and build_agent() factory functions
│   ├── api/
│   │   ├── main.py                  # FastAPI app, routes, lifespan warmup, static mount
│   │   └── models.py                # Pydantic request/response models
│   └── static/
│       └── index.html               # Chat UI (served at /)
├── tests/
│   ├── test_models.py               # Unit tests for Pydantic models (no external services)
│   ├── test_ingestion.py            # (stub)
│   └── test_retrieval.py            # (stub)
├── data/
│   ├── raw/trials.jsonl             # Raw trial records from ClinicalTrials.gov
│   └── enriched/trials_enriched.jsonl  # Claude-enriched records
├── scripts/
│   └── setup_gcp.sh                 # One-shot GCP bootstrap: APIs, Artifact Registry, SA, IAM, Secret Manager
├── .github/workflows/
│   └── deploy.yml                   # CI/CD pipeline
├── Dockerfile                       # Multi-stage build
├── .dockerignore                    # Excludes .env, data/, dev artifacts
├── requirements.txt
├── .env.example                     # Committed template — never commit .env
└── CLAUDE.md
```

---

## Key Implementation Details

### Data Ingestion (`src/ingestion/fetch_trials.py`)

- Queries the ClinicalTrials.gov v2 REST API with a condition filter (e.g. `schizophrenia`)
- Uses `curl_cffi` with `impersonate="chrome"` to bypass TLS fingerprinting that blocks Python's default ssl module
- Paginates via `nextPageToken`; collects up to `--max-results` records (default 200)
- Extracts 8 fields per study: `nctId`, `briefTitle`, `phase`, `conditions`, `interventions`, `eligibilityCriteria`, `overallStatus`, `startDate`
- Writes to `data/raw/trials.jsonl` (one JSON object per line)
- CLI: `python3 -m src.ingestion.fetch_trials "schizophrenia" --max-results 200`

### Enrichment Pipeline (`src/enrichment/enrich_trial.py`)

- Reads `data/raw/trials.jsonl`, skips already-processed records (resumable)
- Calls Anthropic API concurrently with `asyncio.Semaphore(5)` and `asyncio.Lock` for safe file writes
- System prompt uses `cache_control: ephemeral` for prompt caching across calls
- Extracts 8 fields per trial via structured JSON:
  - `inclusion_criteria` — list of strings
  - `exclusion_criteria` — list of strings
  - `age_range` — e.g. `"18-65 years"`
  - `population_descriptor` — plain-language summary of who qualifies
  - `phase_normalized` — e.g. `"Phase 2"`
  - `eligibility_complexity_score` — integer 1–5
  - `plain_language_summary` — 2–3 sentence lay summary
  - `primary_condition_tags` — list of lowercase condition strings
- Writes to `data/enriched/trials_enriched.jsonl`
- CLI: `python3 -m src.enrichment.enrich_trial`

### Embedding & Upsert (`src/embedding/embed_and_upsert.py`)

- Reads enriched JSONL; builds text chunks: `briefTitle + plain_language_summary + "Inclusion: ..." + "Exclusion: ..."`
- Embeds locally using `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dims) — no API key required; model downloads ~130 MB on first run
- Creates Pinecone serverless index if missing (`cloud="aws"`, `region="us-east-1"`, cosine similarity, dim=384)
- Detects dimension mismatch and recreates index automatically
- Upserts in batches of 50; stores chunk text in `metadata["text"]` so `langchain_pinecone` can reconstruct `Document.page_content`
- Pinecone metadata fields per vector:
  - `text` — full chunk text (required by `langchain_pinecone` for `page_content`)
  - `nct_id`, `brief_title`
  - `phase_normalized`, `overall_status`
  - `age_range`, `eligibility_complexity_score`
  - `primary_condition_tags` — list field; filtered with `$in`
- CLI: `python3 -m src.embedding.embed_and_upsert`

### LangChain Chain & Agent (`src/agent/chain.py`)

- `build_qa_chain()` — LCEL chain: `RunnablePassthrough.assign(context, source_documents)` → prompt → Claude → `StrOutputParser`; invoke with `{"question": ...}`; returns `{"answer": str, "source_documents": list[Document]}`
- `build_agent()` — LangGraph-backed `create_agent` with two tools:
  - `search_trials(query)` — semantic similarity search over all records
  - `filter_trials(phase, condition_tag)` — metadata-filtered search using `$eq` for phase and `$in` for condition tags
- Both singletons are `@lru_cache(maxsize=1)` and warmed up at lifespan startup via `run_in_executor`
- LangSmith instrumentation: maps `LANGSMITH_API_KEY` → `LANGCHAIN_API_KEY` + sets `LANGCHAIN_TRACING_V2=true` at import time

### RAG Prompt Design

The system prompt instructs Claude to:
1. Cite the NCT ID for every factual claim, e.g. `(NCT12345678)`
2. Respond with `"I don't have enough information in the available trial records to answer this question."` if context is insufficient
3. Never use knowledge outside the provided context

### API (`src/api/main.py`, `src/api/models.py`)

- `POST /query` — single-turn; routes to agent if filters are set, otherwise uses QA chain
- `POST /chat` — multi-turn; accepts full conversation history as `list[ChatMessage]`; always uses the agent; appends filter hint to last user message
- `GET /health` — pings Pinecone `describe_index_stats`; returns `{"status": "ok", "pinecone_connected": bool}`
- All LangChain calls wrapped in `asyncio.run_in_executor` to avoid blocking the event loop
- `StaticFiles` mounted at `/` **after** all API routes so `/query`, `/chat`, `/health` take precedence
- CORS enabled for `*` (restrict in production)

### Chat UI (`src/static/index.html`)

- Single-file vanilla HTML/CSS/JS; served by FastAPI at `/`
- Maintains conversation history client-side; sends full history on every `/chat` call (stateless server)
- Features: welcome screen with suggested questions, typing indicator, source chips (link to `clinicaltrials.gov/study/<nct_id>`), phase/condition filters in header, Enter to send, Clear button

---

## Development Workflow

### Python Version
Always use **Python 3.11**. Do not use syntax or stdlib features introduced after 3.11.

### FastAPI: Always Async
All route handlers must be `async def`. Wrap synchronous LangChain calls in `run_in_executor`:

```python
result = await asyncio.get_event_loop().run_in_executor(
    None, lambda: chain.invoke({"question": question})
)
```

### Secrets
- Never hardcode API keys anywhere in source code or tests
- All secrets in `.env` (gitignored); `.env.example` committed with placeholders
- In production, secrets are stored in GCP Secret Manager and mounted by Cloud Run via `--update-secrets`

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run ingestion (fetch 200 schizophrenia trials)
python3 -m src.ingestion.fetch_trials "schizophrenia" --max-results 200

# Enrich trials (requires ANTHROPIC_API_KEY in .env)
python3 -m src.enrichment.enrich_trial

# Embed and upsert to Pinecone (requires PINECONE_API_KEY in .env)
python3 -m src.embedding.embed_and_upsert

# Start dev server on port 8000
python3 -m uvicorn src.api.main:app --port 8000 --reload

# Run tests
python3 -m pytest tests/ -v

# Build Docker image
docker build -t clinicalrag:dev .

# Run Docker container (injects .env at runtime — never baked into image)
docker run --rm -p 8080:8080 --env-file .env clinicalrag:dev
```

---

## Deployment Pipeline

Every push to `main` triggers `.github/workflows/deploy.yml`:

```
push to main
  └── pytest (ubuntu-latest, Python 3.11)
        └── docker build + push → europe-west3-docker.pkg.dev/clinicalrag-project/clinicalrag/app:<sha>
              └── gcloud run deploy clinicalrag-prod
                    --region europe-west3
                    --min-instances 0 --max-instances 3 --memory 1Gi
                    --update-secrets ANTHROPIC_API_KEY=...,PINECONE_API_KEY=...,...
```

**Required GitHub secrets:**

| Secret | Value |
|---|---|
| `GOOGLE_CREDENTIALS` | GCP service account JSON key |
| `GCP_PROJECT_ID` | `clinicalrag-project` |

**GCP setup** (one-time): run `bash scripts/setup_gcp.sh` after `gcloud auth login`. The script enables APIs, creates the Artifact Registry repo, service account with IAM roles, and uploads all API keys to Secret Manager from `.env`.

**Live URL:** `https://clinicalrag-prod-280066515559.europe-west3.run.app`

---

## Known Constraints and Next Steps

- **Dataset size**: currently 200 schizophrenia trials. To scale, re-run `fetch_trials.py` with additional conditions and increase `--max-results`. The embedding pipeline handles arbitrary dataset size — just re-run `embed_and_upsert.py`.
- **Retrieval quality**: grounding is tested manually via LangSmith traces. A formal eval harness (e.g. RAGAS) measuring faithfulness and answer relevance is a logical next step.
- **No caching layer**: identical queries re-embed and re-retrieve every time. Add Redis or an in-memory LRU cache keyed on `(question, filters)` for production use.
- **Cold start latency**: with `min-instances=0`, the first request after idle takes ~14s (fastembed model load + Pinecone connection). Set `min-instances=1` if consistent sub-2s latency is required (costs ~$10/month).
- **Single index**: all conditions share one Pinecone index. For multi-condition scale, consider namespaces per condition.

---

## Debugging and Troubleshooting

**API key errors at startup or query time**
- Verify `.env` contains all required keys (see `.env.example`)
- In Cloud Run: check Secret Manager versions are `latest` and the runtime service account has `roles/secretmanager.secretAccessor`
- Watch for trailing newlines in secrets — use `printf '%s'` not `echo` when creating Secret Manager versions

**`pinecone_connected: false` on `/health`**
- Check `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are correct
- Verify the index exists: `pc.list_indexes()` in a Python shell
- Pinecone serverless is regional — ensure the index region matches your client config

**Enrichment slowness or 429 rate limit errors**
- New Anthropic accounts are limited to ~8,000 output tokens/min
- The pipeline is resumable: re-run `enrich_trial.py` and it skips already-processed records
- The semaphore is set to 5 concurrent calls (`src/enrichment/enrich_trial.py:asyncio.Semaphore(5)`); reduce to 2–3 if hitting rate limits consistently

**`Found document with no 'text' key` warnings**
- `langchain_pinecone` uses `metadata["text"]` for `Document.page_content`
- This field is written by `embed_and_upsert.py`; if missing, re-run the upsert to overwrite vectors with the correct metadata

**Docker build fails on `requirements.txt`**
- The builder stage runs `pip install --prefix=/install` — if a package fails, the error appears in the `RUN` layer
- Run `docker build --progress=plain` to see the full pip output

**GitHub Actions deploy fails with `Missing required argument [--image]`**
- Caused by `GCP_PROJECT_ID` secret being empty or unset — the image tag resolves to `europe-west3-docker.pkg.dev//clinicalrag/app:<sha>` (double slash)
- Verify both `GOOGLE_CREDENTIALS` and `GCP_PROJECT_ID` are set in repository secrets
