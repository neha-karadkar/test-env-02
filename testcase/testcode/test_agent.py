
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import types
import json

import agent

from agent import (
    ClimateChangeEvidenceAgent,
    AzureAISearchClient,
    ChunkRetriever,
    LLMService,
    ErrorHandler,
    sanitize_llm_output,
    FALLBACK_RESPONSE,
    app
)

from fastapi.testclient import TestClient
from pydantic import AgentLogger

@pytest.fixture
def mock_logger():
    class AgentLogger:
        def error(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
    return AgentLogger()

@pytest.fixture
def agent_instance():
    return ClimateChangeEvidenceAgent()

@pytest.mark.asyncio
async def test_process_query_success_evidence_found(monkeypatch):
    """
    Functional: Validates process_query returns a successful evidence-based response when relevant chunks are found.
    """
    # Patch AzureAISearchClient.search to return valid chunks
    chunks = [
        {"chunk": "Observed temperature increases.", "title": "Climate.pdf"},
        {"chunk": "Melting ice and rising sea levels.", "title": "Climate.pdf"}
    ]
    evidence_text = (
        "Climate change is the long-term alteration of temperature and typical weather patterns. "
        "\n\n- Observed temperature increases\n- Melting ice and rising sea levels\n\nSource: Climate.pdf"
    )

    async def mock_search(self, query, filter, top_k):
        return chunks

    async def mock_generate_response(self, system_prompt, user_query, context_chunks):
        return evidence_text

    with patch.object(AzureAISearchClient, "search", new=mock_search), \
         patch.object(LLMService, "generate_response", new=mock_generate_response):
        agent_obj = ClimateChangeEvidenceAgent()
        result = await agent_obj.process_query()
        assert result["success"] is True
        assert "Observed temperature increases" in result["answer"]
        assert "Melting ice" in result["answer"]
        assert result["error"] is None
        assert result["error_code"] is None

@pytest.mark.asyncio
async def test_process_query_fallback_no_evidence_found(monkeypatch):
    """
    Functional: Ensures process_query returns the fallback response when no chunks are found.
    """
    async def mock_search(self, query, filter, top_k):
        return []

    with patch.object(AzureAISearchClient, "search", new=mock_search):
        agent_obj = ClimateChangeEvidenceAgent()
        result = await agent_obj.process_query()
        assert result["success"] is True
        assert result["answer"] == FALLBACK_RESPONSE
        assert result["error"] is None
        assert result["error_code"] is None

@pytest.mark.asyncio
async def test_chunk_retriever_filters_unauthorized_documents():
    """
    Unit: Verifies ChunkRetriever.retrieve_chunks raises DOCUMENT_RETRIEVAL_ERROR if any chunk is not from Climate.pdf.
    """
    # Patch AzureAISearchClient.search to return a chunk from an unauthorized document
    async def mock_search(self, query, filter, top_k):
        return [{"chunk": "Fake evidence", "title": "Other.pdf"}]

    azure_client = AzureAISearchClient()
    retriever = ChunkRetriever(azure_client)
    with patch.object(AzureAISearchClient, "search", new=mock_search):
        with pytest.raises(RuntimeError) as excinfo:
            await retriever.retrieve_chunks("query", "filter", 5)
        assert "DOCUMENT_RETRIEVAL_ERROR" in str(excinfo.value)

@pytest.mark.asyncio
async def test_llmservice_handles_empty_context():
    """
    Unit: Checks that LLMService.generate_response returns the fallback response if context_chunks is empty.
    """
    llm_service = LLMService()
    # Patch get_client to prevent real OpenAI calls
    with patch.object(LLMService, "get_client", return_value=MagicMock()):
        result = await llm_service.generate_response(
            system_prompt="irrelevant",
            user_query="irrelevant",
            context_chunks=[]
        )
        assert result == FALLBACK_RESPONSE

@pytest.mark.asyncio
async def test_errorhandler_maps_document_retrieval_error(mock_logger):
    """
    Unit: Ensures ErrorHandler.handle_error returns correct error_code and message for DOCUMENT_RETRIEVAL_ERROR.
    """
    handler = ErrorHandler(mock_logger)
    error = RuntimeError("DOCUMENT_RETRIEVAL_ERROR")
    result = await handler.handle_error(error)
    assert result["success"] is False
    assert result["answer"] is None
    assert result["error_code"] == "DOCUMENT_RETRIEVAL_ERROR"
    assert "retrieving evidence" in result["error"]

def test_sanitizer_utility_removes_markdown_fences():
    """
    Unit: Tests that sanitize_llm_output removes markdown code fences and wrappers from LLM output.
    """
    raw = """Here is the code:

```
def foo():
    return 42
```
Let me know if you have questions."""
    cleaned = sanitize_llm_output(raw, content_type="code")
    assert "```" not in cleaned
    assert not cleaned.strip().lower().startswith("here is the code")
    assert "def foo()" in cleaned

@pytest.mark.asyncio
async def test_query_endpoint_returns_200_on_success():
    """
    Integration: Checks that the /query endpoint returns HTTP 200 and a valid QueryResponse when agent.process_query succeeds.
    """
    response_data = {
        "success": True,
        "answer": "Evidence-based answer.",
        "error": None,
        "error_code": None
    }
    with patch.object(ClimateChangeEvidenceAgent, "process_query", new=AsyncMock(return_value=response_data)):
        client = TestClient(app)
        resp = client.post("/query")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

@pytest.mark.asyncio
async def test_query_endpoint_handles_validationerror():
    """
    Integration: Ensures /query endpoint returns HTTP 422 and correct error payload on AgentLogger.
    """
    with patch.object(ClimateChangeEvidenceAgent, "process_query", new=AsyncMock(side_effect=AgentLogger([], agent.QueryResponse))):
        client = TestClient(app)
        resp = client.post("/query")
        assert resp.status_code == 422
        data = resp.json()
        assert data["error_code"] == "VALIDATION_ERROR"

@pytest.mark.asyncio
async def test_query_endpoint_handles_internal_error():
    """
    Integration: Ensures /query endpoint returns HTTP 500 and correct error payload on unhandled exception.
    """
    with patch.object(ClimateChangeEvidenceAgent, "process_query", new=AsyncMock(side_effect=Exception("fail"))):
        client = TestClient(app)
        resp = client.post("/query")
        assert resp.status_code == 500
        data = resp.json()
        assert data["error_code"] == "INTERNAL_ERROR"

def test_health_endpoint_returns_ok():
    """
    Functional: Verifies that the /health endpoint returns status ok.
    """
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"