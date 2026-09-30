import json

import numpy as np
import pytest

from helpers.helpers import (
    base_chunk,
    cosine_similarity,
    parse_json,
    semantic_chunk,
    tokenize,
)


class TestCosineSimilarity:
    # scores must be plain floats so they can cross the JSON/gRPC boundary
    def test_returns_plain_float(self):
        score = cosine_similarity(
            np.array([1, 0], np.float32), np.array([1, 1], np.float32)
        )
        assert type(score) is float
        json.dumps({"score": score})

    def test_zero_vector_scores_zero(self):
        assert cosine_similarity(np.zeros(2), np.array([1.0, 1.0])) == 0.0


class TestSemanticChunk:
    # line-based journal entries without punctuation must not become one chunk
    def test_splits_on_newlines(self):
        assert semantic_chunk("one\ntwo\nthree\nfour\nfive", 4, 1) == [
            "one two three four",
            "four five",
        ]

    def test_splits_on_sentence_ends_and_blank_lines(self):
        assert semantic_chunk("A b. C d! E?\n\nF g", 2, 0) == ["A b. C d!", "E? F g"]

    def test_blank_text_has_no_chunks(self):
        assert semantic_chunk("  \n ", 4, 1) == []


class TestBaseChunkGuard:
    @pytest.mark.parametrize(
        ("size", "overlap"), [(0, 0), (4, 4), (4, 5), (4, -1), (-1, -2)]
    )
    def test_rejects_invalid_window(self, size, overlap):
        with pytest.raises(ValueError):
            base_chunk(["a", "b", "c"], size, overlap)


class TestParseJson:
    # LLMs often wrap JSON in markdown fences despite the prompt
    @pytest.mark.parametrize(
        "reply",
        [
            '{"status": "found"}',
            '```json\n{"status": "found"}\n```',
            '```\n{"status": "found"}\n```',
            '  ```json {"status": "found"} ```  ',
        ],
    )
    def test_strips_code_fences(self, reply):
        assert parse_json(reply) == {"status": "found"}


class TestBaseChunk:
    @pytest.mark.parametrize(
        ("size", "overlap", "expected"),
        [
            (4, 0, ["a b c d", "e f g"]),
            (4, 1, ["a b c d", "d e f g"]),
            (3, 1, ["a b c", "c d e", "e f g"]),
            (10, 2, ["a b c d e f g"]),
        ],
    )
    def test_windows(self, size, overlap, expected):
        assert base_chunk(list("abcdefg"), size, overlap) == expected

    def test_empty_input_has_no_chunks(self):
        assert base_chunk([], 4, 1) == []


class TestTokenize:
    def test_lowercases_and_drops_stopwords(self):
        assert tokenize("The Cat sat on THE mat") == ["cat", "sat", "mat"]

    # R31 regression: punctuation tokens made every document with a "?" or "." match
    def test_drops_punctuation(self):
        assert tokenize("What made me anxious? Nothing.") == [
            "made",
            "anxious",
            "nothing",
        ]
