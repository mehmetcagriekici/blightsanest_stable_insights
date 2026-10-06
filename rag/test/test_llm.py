from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from botocore.exceptions import ClientError
from ollama import ResponseError

from llm import bedrock, ollama_provider
from llm.bedrock import llm_bedrock
from llm.ollama_provider import llm_ollama

HOST = "http://ollama:11434"
BEDROCK = {"region": "eu-west-1", "model_id": "test-model"}


class TestOllama:
    @pytest.fixture
    def chat(self, monkeypatch) -> AsyncMock:
        chat = AsyncMock()
        client_class = Mock(return_value=SimpleNamespace(chat=chat))
        monkeypatch.setattr(ollama_provider, "AsyncClient", client_class)
        # expose the class mock so tests can check the host it was given
        chat.client_class = client_class
        return chat

    @pytest.mark.asyncio
    async def test_returns_message_content(self, chat):
        chat.return_value = SimpleNamespace(message=SimpleNamespace(content="hi"))

        assert await llm_ollama("user", "system", host=HOST, model="m") == "hi"

        chat.client_class.assert_called_once_with(host=HOST)
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
        assert await llm_ollama("user", "system", host=HOST, model="m") is None


class TestBedrock:
    @pytest.fixture
    def client(self, monkeypatch) -> Mock:
        client = Mock()
        client_factory = Mock(return_value=client)
        monkeypatch.setattr(bedrock.boto3, "client", client_factory)
        # expose the factory so tests can check the region it was given
        client.factory = client_factory
        return client

    @pytest.mark.asyncio
    async def test_returns_text_and_sends_system_prompt_separately(self, client):
        client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hi"}]}}
        }

        assert await llm_bedrock("user", "system", **BEDROCK) == "hi"

        client.factory.assert_called_once_with(
            "bedrock-runtime", region_name="eu-west-1"
        )
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
        assert await llm_bedrock("user", "system", **BEDROCK) is None

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
        assert await llm_bedrock("user", "system", **BEDROCK) is None
