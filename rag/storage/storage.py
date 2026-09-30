import logging

import msgpack
import redis
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from redis.exceptions import RedisError

from custom_types.custom_types import Document, User
from type_converter.type_converter import TypeConverter

logger = logging.getLogger(__name__)

# error codes s3 returns for a key that does not exist. without the
# s3:ListBucket permission s3 answers AccessDenied instead, which is
# (correctly) treated as a failure, not as "missing"
_MISSING_KEY_CODES = {"NoSuchKey", "404"}


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

    # upload data. cache=False skips redis entirely: use it for mutable keys
    # (like the snapshot manifest) that must always be read fresh from s3
    def upload_data(self, document_name, data, cache: bool = True):
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

        if not cache:
            return

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
            # an older value may still be cached; drop it so reads fall
            # through to s3 instead of returning data s3 no longer holds
            self._drop_cached(document_name)

    # load from redis or aws. returns None only when the object does not
    # exist; any other s3 failure raises, so callers never mistake an outage
    # for "not built yet"
    def load_data(self, document_name, cache: bool = True):
        if cache:
            cached = self._load_cached(document_name)
            if cached is not None:
                return cached

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self._key(document_name),
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in _MISSING_KEY_CODES:
                return None
            raise
        return self.type_converter.deserialize(response["Body"].read())

    # delete from s3 and redis. s3 failures raise; redis is best-effort
    def delete_data(self, document_name):
        self.s3_client.delete_object(
            Bucket=self.bucket_name,
            Key=self._key(document_name),
        )
        self._drop_cached(document_name)

    # read and decode a cached value; any redis or decoding failure is
    # treated as a cache miss so the caller falls back to s3
    def _load_cached(self, document_name):
        try:
            cached_data = self.redis_connection.get(self._key(document_name))
        except RedisError as e:
            logger.error("redis object loading failed; falling back to s3: %s", e)
            return None
        if cached_data is None:
            return None
        try:
            return self.type_converter.deserialize(cached_data)
        except (msgpack.UnpackException, ValueError, TypeError, KeyError) as e:
            logger.error("cached object is unreadable; falling back to s3: %s", e)
            return None

    def _drop_cached(self, document_name):
        try:
            self.redis_connection.delete(self._key(document_name))
        except RedisError as e:
            logger.error("redis object delete failed: %s", e)
