import os

# semantic_index loads its model at import time from this variable, so it
# must be set before any test module imports it; an exported value wins
os.environ.setdefault("SENTENCE_TRANSFORMERS_MODEL_NAME", "all-MiniLM-L6-v2")

import pytest

from custom_types.custom_types import Document, User


@pytest.fixture
def mock_user():
    """Create a test user"""
    return User(id="test_user")


@pytest.fixture
def mock_documents():
    """Create test journal entries"""
    return [
        Document(id="doc1", content="Today I felt anxious about my presentation"),
        Document(id="doc2", content="I slept well and felt energized"),
        Document(id="doc3", content="Had a productive meeting with the team"),
        Document(id="doc4", content="Struggled with focus today"),
    ]
