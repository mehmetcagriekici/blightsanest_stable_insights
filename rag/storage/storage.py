import logging

import redis
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from redis.exceptions import RedisError

from custom_types.custom_types import Document, User
from type_converter.type_converter import TypeConverter

logger = logging.getLogger(__name__)


# uploading and loading data to aws and redis
class Storage:
    # the s3 and redis clients are created once (see storage/clients.py)
    # and shared; Storage only scopes them to one user and one bucket
    def __init__(
        self,
        database_user: User,
        bucket_name: str,
        s3_client: BaseClient,
        redis_connection: redis.Redis,
        redis_ttl: int = 3600,
    ) -> None:
        self.database_user: User = database_user
        self.bucket_name = bucket_name
        self.s3_client = s3_client
        self.redis_connection = redis_connection
        self.redis_ttl = redis_ttl

        # type converter for packaging and unpackaging
        self.type_converter = TypeConverter()
        # register pydantic types
        self.type_converter.register_pydantic_models(Document)
        self.type_converter.register_pydantic_models(User)

    # every object for a user lives under users/{user_id}/
    def _key(self, document_name: str) -> str:
        return f"users/{self.database_user.id}/{document_name}"

    # upload data
    def upload_data(self, document_name, data):
        # serialize data
        serialized_data = self.type_converter.serialize(data)
        if not serialized_data:
            raise ValueError("No serialized data to save")

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self._key(document_name),
                Body=serialized_data,
            )
        except ClientError as e:
            raise ClientError(
                {"Error": e.response.get("Error", {})},
                operation_name="aws put object failed",
            ) from e

        # save to redis if aws succeeds - redis is an optional cache, so a
        # redis failure (connection down, timeout, etc.) must not fail the
        # upload now that s3 (the source of truth) has already succeeded
        try:
            self.redis_connection.set(
                name=self._key(document_name),
                value=serialized_data,
                ex=self.redis_ttl,
            )
        except RedisError as e:
            logger.error("redis object upload failed; continuing with s3 only: %s", e)

    # load from redis or aws
    def load_data(self, document_name):
        # redis is an optional cache; if it is unreachable fall back to s3
        try:
            cached_data = self.redis_connection.get(self._key(document_name))
        except RedisError as e:
            logger.error("redis object loading failed; falling back to s3: %s", e)
            cached_data = None

        if cached_data is not None:
            return self.type_converter.deserialize(cached_data)
        else:
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=self._key(document_name),
                )
                content = response["Body"].read()
                return self.type_converter.deserialize(content)
            except ClientError as e:
                logger.error(f"aws object loading failed: {e}")
                return None
