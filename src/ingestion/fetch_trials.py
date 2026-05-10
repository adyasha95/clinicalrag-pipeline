"""Fetch clinical trial records from the ClinicalTrials.gov v2 API."""

import argparse
import asyncio
import json
from pathlib import Path

from curl_cffi.requests import AsyncSession
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_API_BASE = "https://clinicaltrials.gov/api/v2/studies"
_PAGE_SIZE = 100
_OUTPUT_PATH = Path("data/raw/trials.jsonl")


def _extract(study: dict) -> dict:
    """Pull the fields we care about from a raw study object."""
    proto = study.get("protocolSection", {})

    id_module = proto.get("identificationModule", {})
    status_module = proto.get("statusModule", {})
    design_module = proto.get("designModule", {})
    cond_module = proto.get("conditionsModule", {})
    arms_module = proto.get("armsInterventionsModule", {})
    elig_module = proto.get("eligibilityModule", {})

    phases = design_module.get("phases", [])

    interventions = [
        i.get("name", "")
        for i in arms_module.get("interventions", [])
        if i.get("name")
    ]

    return {
        "nctId": id_module.get("nctId"),
        "briefTitle": id_module.get("briefTitle"),
        "phase": phases[0] if phases else None,
        "conditions": cond_module.get("conditions", []),
        "interventions": interventions,
        "eligibilityCriteria": elig_module.get("eligibilityCriteria"),
        "overallStatus": status_module.get("overallStatus"),
        "startDate": status_module.get("startDateStruct", {}).get("date"),
    }


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _fetch_page(
    session: AsyncSession,
    condition: str,
    next_page_token: str | None,
) -> dict:
    params: dict[str, str | int] = {
        "query.cond": condition,
        "pageSize": _PAGE_SIZE,
        "format": "json",
    }
    if next_page_token:
        params["pageToken"] = next_page_token

    response = await session.get(_API_BASE, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


async def fetch_trials(condition: str, max_results: int) -> None:
    """Fetch up to *max_results* studies for *condition* and write to JSONL."""
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fetched = 0
    next_page_token: str | None = None

    async with AsyncSession(impersonate="chrome") as session:
        with _OUTPUT_PATH.open("w", encoding="utf-8") as fh:
            while fetched < max_results:
                try:
                    data = await _fetch_page(session, condition, next_page_token)
                except Exception as exc:
                    print(f"Error after retries — aborting: {exc}")
                    break

                studies = data.get("studies", [])
                if not studies:
                    break

                for study in studies:
                    if fetched >= max_results:
                        break
                    record = _extract(study)
                    fh.write(json.dumps(record) + "\n")
                    fetched += 1
                    if fetched % 100 == 0:
                        print(f"  {fetched} records written...")

                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break

    print(f"Done. {fetched} records written to {_OUTPUT_PATH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch clinical trials from ClinicalTrials.gov"
    )
    parser.add_argument(
        "condition",
        help='Disease or condition to search for, e.g. "schizophrenia"',
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum number of studies to fetch (default: 500)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(fetch_trials(args.condition, args.max_results))
