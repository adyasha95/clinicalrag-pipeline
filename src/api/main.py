"""FastAPI application entry point."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

load_dotenv()

from src.agent.chain import build_agent, build_qa_chain  # noqa: E402
from src.api.models import (  # noqa: E402
    ChatRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TrialSource,
)

logger = logging.getLogger(__name__)

# ── Cached singletons (built once at startup) ─────────────────────────────────

@lru_cache(maxsize=1)
def _qa_chain():
    return build_qa_chain()


@lru_cache(maxsize=1)
def _agent():
    return build_agent()


# ── Lifespan: warm up connections before serving traffic ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up QA chain and agent...")
    loop = asyncio.get_event_loop()
    for name, builder in [("qa_chain", _qa_chain), ("agent", _agent)]:
        try:
            await loop.run_in_executor(None, builder)
            logger.info("%s ready.", name)
        except Exception as exc:
            # Non-fatal — /query will surface the error at request time.
            logger.warning("Failed to warm up %s: %s", name, exc)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ClinicalRAG API",
    description="Retrieval-augmented generation over ClinicalTrials.gov records.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question about clinical trials.

    - If **filters** are supplied (phase and/or condition_tag), the tool-calling
      agent is used so it can invoke `filter_trials` before generating the answer.
    - Otherwise the simpler RetrievalQA chain is used for lower latency.
    """
    t0 = time.monotonic()
    filters = request.filters

    try:
        if filters.phase or filters.condition_tag:
            answer, sources = await _run_agent(request.question, filters.phase, filters.condition_tag)
        else:
            answer, sources = await _run_qa_chain(request.question)
    except Exception as exc:
        logger.exception("Error processing query: %s", request.question)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)


@app.post("/chat", response_model=QueryResponse)
async def chat(request: ChatRequest) -> QueryResponse:
    """Multi-turn chat over clinical trial records.

    Accepts the full conversation history and returns the next assistant turn.
    Filters are applied to every retrieval call throughout the conversation.
    """
    t0 = time.monotonic()
    filters = request.filters

    # Build the message list for the agent.
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Append filter hint to the last user message when filters are active.
    if filters.phase or filters.condition_tag:
        hint_parts = []
        if filters.phase:
            hint_parts.append(f"phase: {filters.phase}")
        if filters.condition_tag:
            hint_parts.append(f"condition: {filters.condition_tag}")
        hint = ", ".join(hint_parts)
        for msg in reversed(messages):
            if msg["role"] == "user":
                msg["content"] = f"{msg['content']} [{hint}]"
                break

    try:
        answer, sources = await _run_agent_messages(messages)
    except Exception as exc:
        logger.exception("Error processing chat")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_ms = int((time.monotonic() - t0) * 1000)
    return QueryResponse(answer=answer, sources=sources, latency_ms=latency_ms)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness check. Pings Pinecone to verify connectivity."""
    pinecone_ok = await _check_pinecone()
    return HealthResponse(status="ok", pinecone_connected=pinecone_ok)


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _run_qa_chain(question: str) -> tuple[str, list[TrialSource]]:
    """Run the LCEL RAG chain in a thread (LangChain is synchronous)."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _qa_chain().invoke({"question": question}),
    )
    answer: str = result.get("answer", "")
    sources = _sources_from_docs(result.get("source_documents", []))
    return answer, sources


async def _run_agent_messages(
    messages: list[dict],
) -> tuple[str, list[TrialSource]]:
    """Run the LangGraph agent with an arbitrary message history."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _agent().invoke({"messages": messages}),
    )
    msgs = result.get("messages", [])
    answer = msgs[-1].content if msgs else ""
    sources = _sources_from_messages(msgs)
    return answer, sources


async def _run_agent(
    question: str, phase: str, condition_tag: str
) -> tuple[str, list[TrialSource]]:
    """Run the LangGraph agent in a thread."""
    filter_hint = ", ".join(
        part
        for part in [
            f"phase: {phase}" if phase else "",
            f"condition: {condition_tag}" if condition_tag else "",
        ]
        if part
    )
    enriched = f"{question} [{filter_hint}]" if filter_hint else question

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _agent().invoke({"messages": [{"role": "user", "content": enriched}]}),
    )
    # LangGraph returns messages list; last message is the assistant reply.
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else ""
    sources = _sources_from_messages(messages)
    return answer, sources


def _sources_from_docs(docs: list) -> list[TrialSource]:
    """Extract TrialSource records from RetrievalQA source_documents."""
    seen: set[str] = set()
    sources: list[TrialSource] = []
    for doc in docs:
        nct_id = doc.metadata.get("nct_id", "")
        if nct_id and nct_id not in seen:
            seen.add(nct_id)
            sources.append(
                TrialSource(
                    nct_id=nct_id,
                    title=doc.metadata.get("brief_title", doc.page_content[:80]),
                )
            )
    return sources


def _sources_from_messages(messages: list) -> list[TrialSource]:
    """Extract TrialSource records from LangGraph agent tool message content."""
    seen: set[str] = set()
    sources: list[TrialSource] = []
    for msg in messages:
        # Tool messages carry the raw string returned by each tool.
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        for block in content.split("---"):
            block = block.strip()
            if not block.startswith("[NCT"):
                continue
            first_line = block.splitlines()[0]
            nct_id = first_line.split("]")[0].lstrip("[")
            lines = block.splitlines()
            title = lines[1][:120].strip() if len(lines) > 1 else ""
            if nct_id and nct_id not in seen:
                seen.add(nct_id)
                sources.append(TrialSource(nct_id=nct_id, title=title))
    return sources


# ── Static UI — mounted last so API routes take precedence ────────────────────
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.is_dir():
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")


async def _check_pinecone() -> bool:
    """Return True if the Pinecone index is reachable."""
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, index.describe_index_stats)
        return True
    except Exception:
        logger.warning("Pinecone health check failed", exc_info=True)
        return False
