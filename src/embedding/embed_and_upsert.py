"""Embed enriched trial chunks and upsert them into Pinecone."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

ENRICHED_PATH = Path("data/enriched/trials_enriched.jsonl")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 64    # fastembed processes locally — generous batch size is fine
UPSERT_BATCH_SIZE = 50   # Pinecone recommended upsert batch size


# ---------------------------------------------------------------------------
# Config — all values sourced from .env, no defaults for secrets
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        print(f"ERROR: {key} is not set. Add it to .env.", file=sys.stderr)
        sys.exit(1)
    return value


# ---------------------------------------------------------------------------
# Text chunk builder
# ---------------------------------------------------------------------------

def _build_chunk(record: dict) -> str:
    """Combine key fields into a single string for embedding."""
    parts: list[str] = []

    title = (record.get("briefTitle") or "").strip()
    if title:
        parts.append(title)

    summary = (record.get("plain_language_summary") or "").strip()
    if summary:
        parts.append(summary)

    inclusion = record.get("inclusion_criteria") or []
    if inclusion:
        parts.append("Inclusion: " + "; ".join(str(c).strip() for c in inclusion if c))

    exclusion = record.get("exclusion_criteria") or []
    if exclusion:
        parts.append("Exclusion: " + "; ".join(str(c).strip() for c in exclusion if c))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Pinecone vector record builder
# ---------------------------------------------------------------------------

def _build_vector(record: dict, embedding: list[float], chunk: str) -> dict:
    """Return a Pinecone upsert record for one trial."""
    tags = record.get("primary_condition_tags") or []

    return {
        "id": record["nctId"],
        "values": embedding,
        "metadata": {
            # langchain_pinecone uses 'text' to populate Document.page_content
            "text": chunk[:8000],
            "nct_id": record.get("nctId", ""),
            "brief_title": record.get("briefTitle", ""),
            "phase_normalized": record.get("phase_normalized", "Unknown"),
            "age_range": record.get("age_range", "Unknown"),
            "eligibility_complexity_score": int(
                record.get("eligibility_complexity_score") or 0
            ),
            "primary_condition_tags": tags if isinstance(tags, list) else [],
            "overall_status": record.get("overallStatus", ""),
        },
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def embed_and_upsert() -> None:
    pinecone_key = _require("PINECONE_API_KEY")
    index_name = _require("PINECONE_INDEX_NAME")

    # ── Read enriched records ──────────────────────────────────────────────
    records: list[dict] = []
    with ENRICHED_PATH.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] line {lineno}: JSON error — {exc}", file=sys.stderr)

    if not records:
        print("No enriched records found. Run enrich_trial.py first.")
        return

    print(f"Loaded {len(records)} enriched records from {ENRICHED_PATH}")

    # ── Connect to Pinecone ────────────────────────────────────────────────
    pc = Pinecone(api_key=pinecone_key)

    existing_indexes = {idx.name for idx in pc.list_indexes()}
    if index_name in existing_indexes:
        # Check if the existing index has the right dimension.
        stats = pc.Index(index_name).describe_index_stats()
        if stats.dimension and stats.dimension != EMBEDDING_DIM:
            print(
                f"Existing index '{index_name}' has dimension {stats.dimension} "
                f"but this run uses {EMBEDDING_DIM}. Deleting and recreating ..."
            )
            pc.delete_index(index_name)
            existing_indexes.discard(index_name)

    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}' (dim={EMBEDDING_DIM}) ...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("  Index created.")
    else:
        print(f"Using existing Pinecone index '{index_name}'.")

    index = pc.Index(index_name)

    # ── Build chunks ───────────────────────────────────────────────────────
    chunks = [_build_chunk(r) for r in records]

    # ── Embed locally with fastembed ───────────────────────────────────────
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads on first run) ...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)

    print(f"Embedding {len(chunks)} chunks ...")
    all_embeddings: list[list[float]] = [
        emb.tolist()
        for emb in model.embed(chunks, batch_size=EMBED_BATCH_SIZE)
    ]
    print(f"  Done — {len(all_embeddings)} embeddings generated.")

    # ── Build Pinecone vectors ─────────────────────────────────────────────
    vectors = [
        _build_vector(record, embedding, chunk)
        for record, embedding, chunk in zip(records, all_embeddings, chunks)
    ]

    # ── Upsert in batches ──────────────────────────────────────────────────
    upsert_batches = [
        vectors[i : i + UPSERT_BATCH_SIZE]
        for i in range(0, len(vectors), UPSERT_BATCH_SIZE)
    ]
    print(
        f"Upserting {len(vectors)} vectors in {len(upsert_batches)} "
        f"batches of {UPSERT_BATCH_SIZE} ..."
    )
    total_upserted = 0
    for batch_idx, batch in enumerate(upsert_batches, 1):
        index.upsert(vectors=batch)
        total_upserted += len(batch)
        print(f"  Upserted batch {batch_idx}/{len(upsert_batches)} "
              f"({total_upserted}/{len(vectors)} total)")

    # ── Final stats ────────────────────────────────────────────────────────
    stats = index.describe_index_stats()
    print(
        f"\nDone. {total_upserted} vectors upserted. "
        f"Index now contains {stats.total_vector_count} vectors."
    )


if __name__ == "__main__":
    embed_and_upsert()
