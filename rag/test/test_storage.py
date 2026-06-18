from botocore.exceptions import ClientError
import pytest
from unittest.mock import Mock, patch
from redis import ResponseError
from storage.storage import Storage

# mock user
@pytest.fixture
def mock_user():
    user = Mock()
    user.id = "user1"
    user.bucket_name = "test_bucket"
    return user

# mock storage
@pytest.fixture
def storage(mock_user):
    with patch("storage.storage.redis.Redis"),\
            patch("storage.storage.boto3.client"):
                # init storage
                s = Storage(mock_user)

                # mock storage attrs
                s.redis_connection = Mock()
                s.s3_client = Mock()
                s.type_converter = Mock()
                s.redis_ttl = 3600

                return s

# test storage initialization
class TestStorageInitialization:
    @patch("storage.storage.redis.Redis")
    @patch("storage.storage.boto3.client")
    def test_registers_models(self, mock_boto, mock_redis, mock_user):
        s = Storage(mock_user)
        assert s.type_converter is not None

# test while uploading data
class TestUploadData:
    # test a successfull upload
    def test_upload_success(self, storage):
        # serialized upload value
        storage.type_converter.serialize.return_value = b"serialized"
        # upload
        storage.upload_data("doc.pkl", {"a": 1})

        # test s3
        storage.s3_client.put_object.assert_called_once_with(
                Bucket="test_bucket",
                Key="user1/doc.pkl",
                Body=b"serialized",
                )

        # test redis
        storage.redis_connection.setex.assert_called_once_with(
                name="user1/doc.pkl",
                time=storage.redis_ttl,
                value=b"serialized",
                )

    # test empty serialization
    # must raise value error
    def test_upload_empty_serializaton(self, storage):
        # empty serialization
        storage.type_converter.serialize.return_value = b""
        with pytest.raises(ValueError):
            storage.upload_data("doc.pkl", {"a": 1})

    # test s3 failure
    def test_upload_s3_failure(self, storage):
        # valid serialized value
        storage.type_converter.serialize.return_value = b"serialized"
        # s3 client error
        storage.s3_client.put_object.side_effect = ClientError(
                {"Error": {"Code": "500"}},
                "PutObject",
                )
        with pytest.raises(ClientError):
            storage.upload_data("doc.pkl", {"a": 1})

        # after s3 failure redis must not be called
        storage.redis_connection.setex.assert_not_called()

    # test for redis failure
    def test_upload_redis_failure(self, storage):
        storage.type_converter.serialize.return_value = b"serialized"
        # redis returns none
        storage.s3_client.put_object.return_value = None

        # init redis error
        storage.redis_connection.setex.side_effect = ResponseError(
                "redis failure"
                )
        with pytest.raises(ResponseError):
            storage.upload_data("doc.pkl", {"a": 1})

# test while loading data
class TestLoadData:
    # test if cache is working
    def test_load_cache_hit(self, storage):
        # cache value
        storage.redis_connection.get.return_value = b"cached"
        # valid deserialized value
        storage.type_converter.deserialize.return_value = {"x": 1}
        # get the result from the storage
        result = storage.load_data("doc.pkl")
        assert result == {"x": 1}
        # result must come from the cache
        storage.s3_client.get_object.assert_not_called()

    # test loading directly from the s3
    def test_load_cache_miss_fallback_to_s3(self, storage):
        # cache is empty
        storage.redis_connection.get.return_value = None
        # creae a mock body to exist in s3
        body = Mock()
        body.read.return_value = b"s3data"
        storage.s3_client.get_object.return_value = {
                "Body": body
                }
        # valid deserialized value
        storage.type_converter.deserialize.return_value = {"x": 1}
        # get the result from the storage
        result = storage.load_data("doc.pkl")
        assert result == {"x": 1}
        # the result must come from the s3
        storage.s3_client.get_object.assert_called_once()

    # test load failure - both cache and s3 -
    def test_load_s3_failure_returns_none(self, storage):
        # no cache value
        storage.redis_connection.get.return_value = None
        # create a client error for s3
        storage.s3_client.get_object.side_effect = ClientError(
                {"Error": {"Code": "NoSuchKey"}},
                "GetObject",
                )
        # get the result from the storage, must be none
        result = storage.load_data("doc.pkl")
        assert result is None
