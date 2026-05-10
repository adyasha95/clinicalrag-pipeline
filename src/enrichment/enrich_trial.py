"""Enrich parsed trial records with additional metadata before embedding."""

import asyncio
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-20250514"
RAW_PATH = Path("data/raw/trials.jsonl")
ENRICHED_PATH = Path("data/enriched/trials_enriched.jsonl")
CONCURRENCY = 5

# Required keys in the LLM's JSON response.
_REQUIRED_FIELDS = {
    "inclusion_criteria",
    "exclusion_criteria",
    "age_range",
    "population_descriptor",
    "phase_normalized",
    "eligibility_complexity_score",
    "plain_language_summary",
    "primary_condition_tags",
}

_SYSTEM_PROMPT = """\
You are a clinical trial eligibility analyst. Given a clinical trial record, \
extract structured metadata from the eligibility criteria and other fields.

Respond with a single JSON object — no markdown fences, no extra text — \
containing exactly these keys:

  inclusion_criteria          – array of strings, one item per distinct \
inclusion criterion parsed from the eligibility text
  exclusion_criteria          – array of strings, one item per distinct \
exclusion criterion
  age_range                   – string describing the eligible age window \
(e.g. "18–65 years", "18+ years", "children 6–17 years", "any age")
  population_descriptor       – 3–5 word phrase naming the target population \
(e.g. "adult treatment-resistant schizophrenia patients")
  phase_normalized            – one of: "Phase 1", "Phase 2", "Phase 3", \
"Phase 4", "Phase 1/2", "Phase 2/3", "N/A", "Unknown"
  eligibility_complexity_score – integer 1–5 where 1 = very simple \
(few broad criteria) and 5 = very complex (many narrow or highly specific criteria)
  plain_language_summary      – 2–3 sentences in plain English describing \
what the trial studies and who may participate
  primary_condition_tags      – array of 2–5 short lowercase strings \
tagging the primary condition(s) and relevant subtypes \
(e.g. ["schizophrenia", "treatment-resistant", "psychosis"])

If a field cannot be determined from the available information, use an \
empty array [] for list fields, "Unknown" for string fields, or 1 for the \
score. Never omit a key.\
"""


def _build_user_message(record: dict) -> str:
    return json.dumps(
        {
            "nctId": record.get("nctId"),
            "briefTitle": record.get("briefTitle"),
            "phase": record.get("phase"),
            "overallStatus": record.get("overallStatus"),
            "conditions": record.get("conditions", []),
            "interventions": record.get("interventions", []),
            "eligibilityCriteria": record.get("eligibilityCriteria") or "",
        },
        indent=2,
    )


def _parse_response(text: str) -> dict:
    """Extract and validate the JSON payload from the model's reply."""
    text = text.strip()
    # Strip optional markdown code fences that some models emit despite instructions.
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.startswith("```")
        ).strip()

    data = json.loads(text)  # raises json.JSONDecodeError on bad output

    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"LLM response missing fields: {missing}")

    score = data["eligibility_complexity_score"]
    if not isinstance(score, int) or not (1 <= score <= 5):
        raise ValueError(
            f"eligibility_complexity_score out of range: {score!r}"
        )

    return data


async def _enrich_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    record: dict,
    write_lock: asyncio.Lock,
    fh,
    counters: dict,
) -> None:
    nct_id = record.get("nctId", "unknown")

    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        # Cache the stable system prompt across all records.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {"role": "user", "content": _build_user_message(record)}
                ],
            )

            raw_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            enrichment = _parse_response(raw_text)
            merged = {**record, **enrichment}

            async with write_lock:
                fh.write(json.dumps(merged) + "\n")
                fh.flush()
                counters["done"] += 1
                done = counters["done"]
                if done % 10 == 0 or done == counters["total"]:
                    print(f"  {done}/{counters['total']} records enriched...")

        except json.JSONDecodeError as exc:
            counters["failed"] += 1
            print(f"  [SKIP] {nct_id}: JSON parse error — {exc}", file=sys.stderr)
        except ValueError as exc:
            counters["failed"] += 1
            print(f"  [SKIP] {nct_id}: validation error — {exc}", file=sys.stderr)
        except anthropic.APIError as exc:
            counters["failed"] += 1
            print(f"  [SKIP] {nct_id}: API error — {exc}", file=sys.stderr)


def _load_existing_ids() -> set[str]:
    """Return the set of nctIds already written to the enriched output file."""
    ids: set[str] = set()
    if not ENRICHED_PATH.exists():
        return ids
    with ENRICHED_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["nctId"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def _read_pending(existing_ids: set[str]) -> list[dict]:
    """Read raw records, skipping any already enriched."""
    pending: list[dict] = []
    with RAW_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("nctId") not in existing_ids:
                pending.append(rec)
    return pending


async def enrich_trials() -> None:
    """Read raw trials, enrich each via the Anthropic API, write enriched output."""
    existing_ids = _load_existing_ids()
    if existing_ids:
        print(f"Resuming — skipping {len(existing_ids)} already-enriched records.")

    records = _read_pending(existing_ids)
    if not records:
        print("Nothing to enrich.")
        return

    print(f"Enriching {len(records)} records with up to {CONCURRENCY} concurrent calls...")

    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "failed": 0, "total": len(records)}

    with ENRICHED_PATH.open("a", encoding="utf-8") as fh:
        tasks = [
            _enrich_one(client, semaphore, record, write_lock, fh, counters)
            for record in records
        ]
        await asyncio.gather(*tasks)

    print(
        f"\nDone. {counters['done']} enriched, "
        f"{counters['failed']} skipped. "
        f"Output → {ENRICHED_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(enrich_trials())
