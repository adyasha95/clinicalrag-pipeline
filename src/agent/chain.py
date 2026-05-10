"""LangChain RAG chain and tool-calling agent for ClinicalRAG."""

import os

from dotenv import load_dotenv

load_dotenv()

# ── LangSmith instrumentation ─────────────────────────────────────────────────
# Must be set before any LangChain objects are created.
_langsmith_key = os.environ.get("LANGSMITH_API_KEY")
if _langsmith_key:
    os.environ.setdefault("LANGCHAIN_API_KEY", _langsmith_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.environ.get("LANGCHAIN_PROJECT", "clinicalrag"))

from langchain.agents import create_agent  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_community.embeddings import FastEmbedEmbeddings  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda, RunnablePassthrough  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_pinecone import PineconeVectorStore  # noqa: E402
from pinecone import Pinecone  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5

# ── Prompts ───────────────────────────────────────────────────────────────────

_QA_SYSTEM = """\
You are a clinical trials research assistant. Answer the question using ONLY the \
clinical trial records supplied in the context below.

Rules:
1. Cite the NCT ID for every factual claim, e.g. (NCT12345678).
2. If the context does not contain enough information to answer, respond with exactly:
   "I don't have enough information in the available trial records to answer this question."
3. Do not use knowledge outside the provided context.
4. Be concise and precise.

Context:
{context}"""

_AGENT_SYSTEM = """\
You are a clinical trials research assistant with access to a database of \
ClinicalTrials.gov records. Use the available tools to retrieve relevant trials \
and answer the user's question.

Rules:
1. Always cite the NCT ID when referring to a specific trial, e.g. (NCT12345678).
2. Base your answer only on the trials returned by the tools.
3. If the tools return no useful results, say: \
"I don't have enough information in the available trial records to answer this question."\
"""

# ── Helpers ───────────────────────────────────────────────────────────────────


def _require(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"{key} is not set. Add it to .env.")
    return value


def _get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def _get_vectorstore() -> PineconeVectorStore:
    pc = Pinecone(api_key=_require("PINECONE_API_KEY"))
    return PineconeVectorStore(
        index=pc.Index(_require("PINECONE_INDEX_NAME")),
        embedding=_get_embeddings(),
    )


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=MODEL,
        api_key=_require("ANTHROPIC_API_KEY"),
        max_tokens=2048,
    )


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant trials found."
    parts = []
    for doc in docs:
        m = doc.metadata
        header = (
            f"[{m.get('nct_id', 'unknown')}]"
            f"  Phase: {m.get('phase_normalized', 'N/A')}"
            f"  Status: {m.get('overall_status', 'N/A')}"
            f"  Age range: {m.get('age_range', 'N/A')}"
            f"  Complexity: {m.get('eligibility_complexity_score', 'N/A')}/5"
            f"  Tags: {', '.join(m.get('primary_condition_tags', []))}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Public factory functions ──────────────────────────────────────────────────


def build_qa_chain():
    """Return an LCEL RAG chain backed by Pinecone and Claude.

    Usage::

        chain = build_qa_chain()
        result = chain.invoke("What phase 3 schizophrenia trials allow adolescents?")
        # result is a dict: {"answer": str, "source_documents": list[Document]}
    """
    vs = _get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _QA_SYSTEM),
            ("human", "{question}"),
        ]
    )

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(retriever.invoke(x["question"]))),
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"])),
        )
        | RunnablePassthrough.assign(
            answer=(
                RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
                | prompt
                | llm
                | StrOutputParser()
            )
        )
    )

    return chain


def build_agent():
    """Return a LangChain 1.x agent (LangGraph-backed) with search_trials and filter_trials.

    Usage::

        agent = build_agent()
        result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
        print(result["messages"][-1].content)
    """
    vs = _get_vectorstore()
    llm = _get_llm()

    @tool
    def search_trials(query: str) -> str:
        """Semantic search over all clinical trial records.

        Args:
            query: A natural-language description of the trials to find.
        """
        docs = vs.similarity_search(query, k=TOP_K)
        return _format_docs(docs)

    @tool
    def filter_trials(phase: str, condition_tag: str) -> str:
        """Filter trials by phase and/or condition tag, then rank by relevance.

        Args:
            phase: Trial phase to filter on, e.g. "Phase 3". Pass empty string to skip.
            condition_tag: Condition tag to filter on, e.g. "schizophrenia". Pass empty string to skip.
        """
        metadata_filter: dict = {}
        if phase:
            metadata_filter["phase_normalized"] = {"$eq": phase}
        if condition_tag:
            metadata_filter["primary_condition_tags"] = {"$in": [condition_tag.lower()]}

        query = " ".join(filter(None, [condition_tag, phase, "clinical trial"]))
        docs = vs.similarity_search(
            query,
            k=TOP_K,
            filter=metadata_filter if metadata_filter else None,
        )
        return _format_docs(docs)

    return create_agent(
        model=llm,
        tools=[search_trials, filter_trials],
        system_prompt=_AGENT_SYSTEM,
    )
