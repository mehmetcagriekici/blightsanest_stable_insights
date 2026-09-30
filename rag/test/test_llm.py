from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from botocore.exceptions import ClientError, NoRegionError
from ollama import ResponseError

from llm import bedrock, ollama_provider
from llm.bedrock import llm_bedrock
from llm.ollama_provider import llm_ollama


class TestOllama:
    @pytest.fixture
    def chat(self, monkeypatch) -> AsyncMock:
        chat = AsyncMock()
        client_class = Mock(return_value=SimpleNamespace(chat=chat))
        monkeypatch.setattr(ollama_provider, "AsyncClient", client_class)
        return chat

    @pytest.mark.asyncio
    async def test_returns_message_content(self, chat):
        chat.return_value = SimpleNamespace(message=SimpleNamespace(content="hi"))

        assert await llm_ollama("user", "system", model="m") == "hi"

        kwargs = chat.call_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["messages"] == [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

    # failures return None so RAG.rag's None guard handles them
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            ResponseError("model not found", 404),
            ConnectionError("ollama unreachable"),
            httpx.ReadTimeout("timed out"),
        ],
        ids=["response-error", "connection-error", "http-error"],
    )
    async def test_failures_return_none(self, chat, error):
        chat.side_effect = error
        assert await llm_ollama("user", "system") is None


class TestBedrock:
    @pytest.fixture
    def client(self, monkeypatch) -> Mock:
        client = Mock()
        monkeypatch.setattr(bedrock, "model_id", "test-model")
        monkeypatch.setattr(bedrock.boto3, "client", Mock(return_value=client))
        return client

    @pytest.mark.asyncio
    async def test_returns_text_and_sends_system_prompt_separately(self, client):
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}}
        }

        assert await llm_bedrock("user", "system") == "hi"

        client.converse.assert_called_once_with(
            modelId="test-model",
            messages=[{"role": "user", "content": [{"text": "user"}]}],
            system=[{"text": "system"}],
        )

    @pytest.mark.asyncio
    async def test_client_error_returns_none(self, client):
        client.converse.side_effect = ClientError(
            {"Error": {"Code": "ThrottlingException"}}, "Converse"
        )
        assert await llm_bedrock("user", "system") is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response",
        [
            {},
            {"output": {"message": {"content": []}}},
            {"output": {"message": {"content": [{"reasoningContent": {}}]}}},
        ],
        ids=["no-output", "empty-content", "no-text-block"],
    )
    async def test_malformed_response_returns_none(self, client, response):
        client.converse.return_value = response
        assert await llm_bedrock("user", "system") is None

    # R37 regression: client creation raises NoRegionError without a region,
    # which must return None instead of escaping
    @pytest.mark.asyncio
    async def test_missing_region_returns_none(self, monkeypatch):
        monkeypatch.setattr(bedrock.boto3, "client", Mock(side_effect=NoRegionError()))
        assert await llm_bedrock("user", "system") is None
