from botocore.client import logging
from botocore.exceptions import ClientError
import redis
import boto3

from type_converter.type_converter import TypeConverter
from custom_types.custom_types import User, Document

# uploading and loading data to aws and redis
class Storage:
    def __init__(self, database_user: User, redis_host="localhost", redis_port=6379, redis_ttl=3600) -> None:
        self.redis_ttl = redis_ttl
        self.database_user: User = database_user
        
        # redis connection
        self.redis_connection = redis.Redis(host=redis_host, port=redis_port)

        # aws s3 client
        self.s3_client = boto3.client("s3")

        # type converter for packaging and unpackaging
        self.type_converter = TypeConverter()
        # register pydantic types
        self.type_converter.register_pydantic_models(Document)
        self.type_converter.register_pydantic_models(User)
        

    # upload data
    def upload_data(self, document_name, data):
        # serialize data
        serialized_data = self.type_converter.serialize(data)
        if not serialized_data:
            raise ValueError("No serialized data to save")

        try:
            self.s3_client.put_object(Bucket=self.database_user.bucket_name, Key=f"{self.database_user.id}/{document_name}", Body=serialized_data)
        except ClientError as e:
            raise ClientError({"Error": e.response.get("Error", {})}, operation_name="aws put object failed") from e

        # save to redis if aws succeeds - redis is an optional cache, so a
        # redis failure (connection down, timeout, etc.) must not fail the
        # upload now that s3 (the source of truth) has already succeeded
        try:
            self.redis_connection.setex(name=f"{self.database_user.id}/{document_name}", time=self.redis_ttl, value=serialized_data)
        except redis.exceptions.RedisError as e:
            logging.error("redis object upload failed; continuing with s3 only: %s", e)

    # load from redis or aws
    def load_data(self, document_name):
        # redis is an optional cache; if it is unreachable fall back to s3
        try:
            cached_data = self.redis_connection.get(f"{self.database_user.id}/{document_name}")
        except redis.exceptions.RedisError as e:
            logging.error("redis object loading failed; falling back to s3: %s", e)
            cached_data = None

        if cached_data:
            return self.type_converter.deserialize(cached_data)
        else:
            try:
                response = self.s3_client.get_object(Bucket=self.database_user.bucket_name, Key=f"{self.database_user.id}/{document_name}")
                content = response["Body"].read()
                return self.type_converter.deserialize(content)
            except ClientError as e:
                print(f"aws object loading failed: {e}")
                return None
