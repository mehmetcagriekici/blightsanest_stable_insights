import os

# semantic_index loads its model at import time from this variable, so it
# must be set before any test module imports it; an exported value wins
os.environ.setdefault("SENTENCE_TRANSFORMERS_MODEL_NAME", "all-MiniLM-L6-v2")

import boto3
import pytest
from moto import mock_aws
from redis.exceptions import ConnectionError as RedisConnectionError

from custom_types.custom_types import Document, User
from storage.storage import Storage


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


# in-memory stand-in for redis.Redis with the calls Storage makes. set
# `down = True` to make every call fail the way an unreachable redis does
class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.down = False

    def _check(self) -> None:
        if self.down:
            raise RedisConnectionError("redis is down")

    def get(self, name):
        self._check()
        return self.data.get(name)

    def set(self, name, value, ex=None):
        self._check()
        self.data[name] = value

    def delete(self, *names):
        self._check()
        for name in names:
            self.data.pop(name, None)


@pytest.fixture
def fake_redis() -> FakeRedis:
    return FakeRedis()


# a real Storage over a moto S3 bucket and FakeRedis
@pytest.fixture
def s3_storage(mock_user, fake_redis):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test_bucket")
        yield Storage(mock_user, "test_bucket", s3, fake_redis)
