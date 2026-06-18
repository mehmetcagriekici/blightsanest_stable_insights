import pytest
import boto3
from moto import mock_aws
from redis import Redis
from custom_types.custom_types import User, Document


@pytest.fixture
def mock_s3_bucket():
    """Create a mocked S3 bucket for tests"""
    with mock_aws():
        # Create the bucket
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test_bucket")
        yield s3

@pytest.fixture
def mock_user():
    """Create a test user with mocked AWS credentials"""
    return User(
        id="test_user",
        aws_access_key_id="test_key_id",
        aws_secret_access_key="test_secret",
        region="us-east-1",
        bucket_name="test_bucket",
    )

@pytest.fixture
def mock_documents():
    """Create test journal entries"""
    return [
        Document(id="doc1", content="Today I felt anxious about my presentation"),
        Document(id="doc2", content="I slept well and felt energized"),
        Document(id="doc3", content="Had a productive meeting with the team"),
        Document(id="doc4", content="Struggled with focus today"),
    ]

@pytest.fixture
def redis_connection():
    """Connect to real Redis (assumes Docker Compose running)"""
    r = Redis(host="localhost", port=6379)
    # Flush the test database before test
    r.flushdb()
    yield r
    # Clean up after
    r.flushdb()
