"""Pydantic request and response models for the ClinicalRAG API."""

from typing import Literal

from pydantic import BaseModel, Field


class QueryFilters(BaseModel):
    phase: str = Field(
        default="",
        description="Trial phase to filter on, e.g. 'Phase 3'. Empty string means no filter.",
        examples=["Phase 3"],
    )
    condition_tag: str = Field(
        default="",
        description="Condition tag to filter on, e.g. 'schizophrenia'. Empty string means no filter.",
        examples=["schizophrenia"],
    )


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language question about clinical trials.",
        examples=["What phase 3 trials are studying treatment-resistant schizophrenia?"],
    )
    filters: QueryFilters = Field(
        default_factory=QueryFilters,
        description="Optional metadata filters applied before retrieval.",
    )


class TrialSource(BaseModel):
    nct_id: str = Field(description="ClinicalTrials.gov identifier, e.g. NCT12345678.")
    title: str = Field(description="Brief title of the trial.")


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer grounded in retrieved trial records.")
    sources: list[TrialSource] = Field(
        description="Trials cited or retrieved as context for the answer."
    )
    latency_ms: int = Field(description="End-to-end server-side latency in milliseconds.")


class HealthResponse(BaseModel):
    status: str = Field(description="Always 'ok' when the API is reachable.")
    pinecone_connected: bool = Field(description="Whether Pinecone index is reachable.")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(description="Message author.")
    content: str = Field(description="Message text.")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Full conversation history, ending with the latest user message.",
    )
    filters: QueryFilters = Field(
        default_factory=QueryFilters,
        description="Optional metadata filters applied to every retrieval turn.",
    )
