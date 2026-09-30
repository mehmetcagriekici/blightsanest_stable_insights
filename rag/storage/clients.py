import boto3
import redis
from botocore.client import BaseClient
from redis.backoff import NoBackoff
from redis.retry import Retry

from config.config import Config


# s3 client for the source of truth. no keys are passed: boto3 resolves
# credentials itself (env vars or ~/.aws locally, the pod's IAM role on EKS)
def create_s3_client(config: Config) -> BaseClient:
    return boto3.client("s3", region_name=config.region)


# redis client for the optional hot cache. short timeouts and no retries so
# an unreachable redis fails fast and storage falls back to s3; redis-py
# retries 3 times with backoff by default, which turns a 2s timeout into ~25s
def create_redis_client(config: Config) -> redis.Redis:
    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        socket_connect_timeout=2,
        socket_timeout=2,
        retry=Retry(NoBackoff(), 0),
    )
