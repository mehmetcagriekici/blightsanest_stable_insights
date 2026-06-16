from botocore.client import ClientError, logging
import msgpack
import redis
import boto3
from redis.exceptions import ResponseError

from type_converter.type_converter import TypeConverter
from types.types import User, Document

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
        
    # serialize and deserialize to help with type conversions
    def serialize(self, data):
        # convert data to serialize format
        serialized = self.type_converter.convert_to_serializable(data)
        # accepts list
        return msgpack.packb(serialized)

    def deserialize_data(self, serialized_data):
        # unpack the serialized data
        unpacked = msgpack.unpackb(serialized_data, raw=False)
        # deserialize and return
        return self.type_converter.convert_back_from_serialized(unpacked)

    # upload data
    def upload_data(self, document_name, data):
        # serialize data
        serialized_data = self.serialize(data)
        if not serialized_data:
            raise ValueError("No serialized data to save")

        try:
            self.s3_client.put_object(Bucket=self.database_user.bucket_name, Key=f"{self.database_user.id}/{document_name}", Body=serialized_data)
        except ClientError as e:
            raise ClientError(e, operation_name="aws put object failed")

        # save to redis if aws succeeds
        try:
            self.redis_connection.setex(name=f"{self.database_user.id}/{document_name}", time=self.redis_ttl, value=serialized_data)
        except redis.ResponseError as e:
            raise ResponseError(e, "redis object upload failed")

    # load from redis or aws
    def load_data(self, document_name):
        cached_data = self.redis_connection.get(f"{self.database_user.id}/{document_name}")
        if cached_data:
            return self.deserialize_data(cached_data)
        else:
            try:
                response = self.s3_client.get_object(Bucket=self.database_user.bucket_name, Key=f"{self.database_user.id}/{document_name}")
                content = response["Body"].read()
                return self.deserialize_data(content)
            except ClientError as e:
                logging.error(e, "aws object loading failed")
                return None
