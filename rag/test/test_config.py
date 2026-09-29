import pytest
from moto import mock_aws

from config.config import Config, load_config
from storage.clients import create_redis_client, create_s3_client


class TestLoadConfig:
    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "my-bucket")
        for name in ("AWS_REGION", "REDIS_HOST", "REDIS_PORT"):
            monkeypatch.delenv(name, raising=False)

        assert load_config() == Config(
            bucket_name="my-bucket",
            region="us-east-1",
            redis_host="localhost",
            redis_port=6379,
        )

    def test_reads_environment(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET", "my-bucket")
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        monkeypatch.setenv("REDIS_HOST", "redis")
        monkeypatch.setenv("REDIS_PORT", "6380")

        assert load_config() == Config(
            bucket_name="my-bucket",
            region="eu-central-1",
            redis_host="redis",
            redis_port=6380,
        )

    @pytest.mark.parametrize(
        ("env", "message"),
        [
            ({}, "S3_BUCKET must be set"),
            ({"S3_BUCKET": ""}, "S3_BUCKET must be set"),
            ({"S3_BUCKET": "b", "REDIS_PORT": "abc"}, "invalid REDIS_PORT"),
        ],
    )
    def test_invalid(self, monkeypatch, env, message):
        for name in ("S3_BUCKET", "REDIS_PORT"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        with pytest.raises(ValueError, match=message):
            load_config()


class TestClients:
    def test_s3_client_uses_config_region(self):
        # moto supplies fake credentials, so the test does not depend on
        # the machine's own AWS setup
        with mock_aws():
            client = create_s3_client(Config(bucket_name="b", region="eu-west-1"))
        assert client.meta.region_name == "eu-west-1"

    def test_redis_client_uses_config_and_fails_fast(self):
        client = create_redis_client(
            Config(bucket_name="b", redis_host="redis", redis_port=6380)
        )
        kwargs = client.connection_pool.connection_kwargs
        assert kwargs["host"] == "redis"
        assert kwargs["port"] == 6380
        assert kwargs["socket_connect_timeout"] == 2
        assert kwargs["socket_timeout"] == 2
