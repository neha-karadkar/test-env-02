
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import agent

@pytest.fixture
def client():
    return TestClient(agent.app)

@pytest.mark.asyncio
async def test_functional_query_endpoint_returns_evidence_when_knowledge_base_has_data(client):
    """
    Ensures that the /query endpoint returns a successful evidence-based answer when relevant chunks are available.
    """
    # Patch AzureAISearchClient.search to return relevant chunks
    chunks = [
        {"chunk": "Observed global temperature increases over the past century.", "title": "Climate.pdf"},
        {"chunk": "Melting of polar ice caps and glaciers.", "title": "Climate.pdf"},
    ]
    # Patch LLMService.generate_response to return a plausible evidence-based answer
    llm_answer = (
        "Climate change refers to long-term shifts in temperatures and weather patterns. "
        "Evidence includes:\n"
        "- Global temperature increases\n"
        "- Melting ice\n"
        "- Rising sea levels\n"
        "Source: Climate.pdf"
    )

    # Patch the agent internals to avoid real Azure/OpenAI calls
    with patch.object(agent.AzureAISearchClient, "search", new=AsyncMock(return_value=chunks)), \
         patch.object(agent.LLMService, "generate_response", new=AsyncMock(return_value=llm_answer)), \
         patch.object(agent, "sanitize_llm_output", side_effect=lambda x, content_type="text": x):

        response = client.post("/query")
        assert response.status_code == 200

        # The endpoint returns a Pydantic model, but TestClient returns .json()
        data = response.json()
        assert isinstance(data, dict)
        assert data["success"] is True
        assert isinstance(data["answer"], str)
        assert data["answer"].strip() != ""
        assert agent.FALLBACK_RESPONSE not in data["answer"]
        assert data["error"] is None
        assert data["error_code"] is None