import os
from dataclasses import dataclass


# runtime configuration for the RAG service, loaded once at startup
@dataclass(frozen=True)
class Config:
    bucket_name: str
    region: str = "us-east-1"
    redis_host: str = "localhost"
    redis_port: int = 6379


# build a Config from environment variables, falling back to defaults
# for anything optional
def load_config() -> Config:
    bucket_name = os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("S3_BUCKET must be set")

    port = os.getenv("REDIS_PORT", "6379")
    try:
        redis_port = int(port)
    except ValueError as e:
        raise ValueError(f"invalid REDIS_PORT {port!r}") from e

    return Config(
        bucket_name=bucket_name,
        region=os.getenv("AWS_REGION", "us-east-1"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=redis_port,
    )
