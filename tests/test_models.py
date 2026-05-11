"""Unit tests for API request/response models — no external services required."""

import pytest
from pydantic import ValidationError

from src.api.models import (
    HealthResponse,
    QueryFilters,
    QueryRequest,
    QueryResponse,
    TrialSource,
)


class TestQueryFilters:
    def test_defaults_are_empty_strings(self):
        f = QueryFilters()
        assert f.phase == ""
        assert f.condition_tag == ""

    def test_accepts_valid_values(self):
        f = QueryFilters(phase="Phase 3", condition_tag="schizophrenia")
        assert f.phase == "Phase 3"
        assert f.condition_tag == "schizophrenia"


class TestQueryRequest:
    def test_valid_question(self):
        r = QueryRequest(question="What trials exist for schizophrenia?")
        assert r.question == "What trials exist for schizophrenia?"
        assert r.filters.phase == ""

    def test_with_filters(self):
        r = QueryRequest(
            question="Show me trials",
            filters={"phase": "Phase 2", "condition_tag": "depression"},
        )
        assert r.filters.phase == "Phase 2"
        assert r.filters.condition_tag == "depression"

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_missing_question_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest()


class TestQueryResponse:
    def test_valid_response(self):
        r = QueryResponse(
            answer="Based on the trials...",
            sources=[TrialSource(nct_id="NCT12345678", title="A trial")],
            latency_ms=1234,
        )
        assert r.answer == "Based on the trials..."
        assert len(r.sources) == 1
        assert r.sources[0].nct_id == "NCT12345678"
        assert r.latency_ms == 1234

    def test_empty_sources_allowed(self):
        r = QueryResponse(answer="No results.", sources=[], latency_ms=500)
        assert r.sources == []


class TestHealthResponse:
    def test_healthy(self):
        h = HealthResponse(status="ok", pinecone_connected=True)
        assert h.status == "ok"
        assert h.pinecone_connected is True

    def test_degraded(self):
        h = HealthResponse(status="ok", pinecone_connected=False)
        assert h.pinecone_connected is False
