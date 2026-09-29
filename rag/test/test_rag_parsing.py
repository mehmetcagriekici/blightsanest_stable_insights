import pytest

from rag.rag import RAG


def rag_replying(reply: str | None) -> RAG:
    async def generate(user_prompt: str, system_prompt: str) -> str | None:
        return reply

    return RAG(generate=generate)


class TestRagParsing:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("reply", "status"),
        [
            ('{"status": "found", "response": "x"}', "found"),
            ('{"status": "not found", "response": "x"}', "not found"),
            ('```json\n{"status": "found", "response": "x"}\n```', "found"),
        ],
    )
    async def test_accepts_valid_replies(self, reply, status):
        result = await rag_replying(reply).rag("q", [])
        assert result.status == status
        assert result.response == "x"

    # every invalid reply surfaces as the same ValueError
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply",
        [
            None,
            "not json",
            '"found"',
            '["status", "response"]',
            '{"status": "maybe", "response": "x"}',
            '{"status": "found"}',
        ],
    )
    async def test_rejects_invalid_replies(self, reply):
        with pytest.raises(ValueError):
            await rag_replying(reply).rag("q", [])
