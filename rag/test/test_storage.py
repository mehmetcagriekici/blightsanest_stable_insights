from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError
from redis import ResponseError

from storage.storage import Storage


# mock storage
@pytest.fixture
def storage(mock_user):
    # clients are passed in, so plain mocks replace s3 and redis
    s = Storage(
        mock_user,
        bucket_name="test_bucket",
        s3_client=Mock(),
        redis_connection=Mock(),
    )
    s.type_converter = Mock()
    return s


# test storage initialization
class TestStorageInitialization:
    # the models storage persists must be registered with its converter
    def test_registers_models(self, mock_user):
        s = Storage(mock_user, "test_bucket", Mock(), Mock())
        assert "Document" in s.type_converter.deserializers
        assert "User" in s.type_converter.deserializers


# every object for a user lives under users/{user_id}/
class TestKeys:
    @pytest.mark.parametrize("user_id", ["user1", "user2"])
    def test_key_uses_users_prefix(self, storage, user_id):
        storage.database_user.id = user_id
        assert storage._key("doc.pkl") == f"users/{user_id}/doc.pkl"


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
            Key="users/test_user/doc.pkl",
            Body=b"serialized",
        )

        # test redis
        storage.redis_connection.set.assert_called_once_with(
            name="users/test_user/doc.pkl",
            value=b"serialized",
            ex=storage.redis_ttl,
        )

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
        storage.redis_connection.set.assert_not_called()

    # redis is an optional cache: a redis failure after a successful s3 write
    # must not fail the upload
    def test_upload_redis_failure_is_non_fatal(self, storage):
        storage.type_converter.serialize.return_value = b"serialized"
        # s3 write succeeds
        storage.s3_client.put_object.return_value = None

        # redis raises on set
        storage.redis_connection.set.side_effect = ResponseError("redis failure")
        # should not raise - s3 (source of truth) already succeeded
        storage.upload_data("doc.pkl", {"a": 1})
        storage.s3_client.put_object.assert_called_once()


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
        # the cache is read under the user's prefix
        storage.redis_connection.get.assert_called_once_with("users/test_user/doc.pkl")
        # result must come from the cache
        storage.s3_client.get_object.assert_not_called()

    # test loading directly from the s3
    def test_load_cache_miss_fallback_to_s3(self, storage):
        # cache is empty
        storage.redis_connection.get.return_value = None
        # creae a mock body to exist in s3
        body = Mock()
        body.read.return_value = b"s3data"
        storage.s3_client.get_object.return_value = {"Body": body}
        # valid deserialized value
        storage.type_converter.deserialize.return_value = {"x": 1}
        # get the result from the storage
        result = storage.load_data("doc.pkl")
        assert result == {"x": 1}
        # the result must come from s3, under the same key as the cache
        storage.redis_connection.get.assert_called_once_with("users/test_user/doc.pkl")
        storage.s3_client.get_object.assert_called_once_with(
            Bucket="test_bucket",
            Key="users/test_user/doc.pkl",
        )

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

    # R29 regression: only a missing key means "no data"; any other s3
    # failure must surface instead of looking like an unbuilt index
    @pytest.mark.parametrize("code", ["AccessDenied", "SlowDown", "500"])
    def test_load_other_s3_errors_raise(self, storage, code):
        storage.redis_connection.get.return_value = None
        storage.s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": code}}, "GetObject"
        )
        with pytest.raises(ClientError):
            storage.load_data("doc.pkl")

    # an unreadable cached value is a cache miss, not an error
    def test_load_corrupt_cache_falls_back_to_s3(self, storage):
        storage.redis_connection.get.return_value = b"garbage"
        body = Mock()
        body.read.return_value = b"s3data"
        storage.s3_client.get_object.return_value = {"Body": body}
        storage.type_converter.deserialize.side_effect = [ValueError("bad"), {"x": 1}]

        assert storage.load_data("doc.pkl") == {"x": 1}
        storage.s3_client.get_object.assert_called_once()

    # cache=False reads s3 directly, for mutable keys like the manifest
    def test_load_without_cache_skips_redis(self, storage):
        body = Mock()
        body.read.return_value = b"s3data"
        storage.s3_client.get_object.return_value = {"Body": body}
        storage.type_converter.deserialize.return_value = {"x": 1}

        assert storage.load_data("doc.pkl", cache=False) == {"x": 1}
        storage.redis_connection.get.assert_not_called()


class TestCacheConsistency:
    def test_upload_without_cache_skips_redis(self, storage):
        storage.type_converter.serialize.return_value = b"serialized"
        storage.upload_data("doc.pkl", {"a": 1}, cache=False)
        storage.s3_client.put_object.assert_called_once()
        storage.redis_connection.set.assert_not_called()

    # R27 regression: a failed redis write must not leave an older cached
    # value behind that s3 no longer holds
    def test_failed_redis_write_drops_the_cached_value(self, storage):
        storage.type_converter.serialize.return_value = b"serialized"
        storage.redis_connection.set.side_effect = ResponseError("redis failure")
        storage.upload_data("doc.pkl", {"a": 1})
        storage.redis_connection.delete.assert_called_once_with(
            "users/test_user/doc.pkl"
        )

    # if redis is fully down the drop fails too; that must stay non-fatal
    def test_failed_redis_write_and_drop_is_non_fatal(self, storage):
        storage.type_converter.serialize.return_value = b"serialized"
        storage.redis_connection.set.side_effect = ResponseError("down")
        storage.redis_connection.delete.side_effect = ResponseError("down")
        storage.upload_data("doc.pkl", {"a": 1})

    def test_delete_removes_from_s3_and_redis(self, storage):
        storage.delete_data("doc.pkl")
        storage.s3_client.delete_object.assert_called_once_with(
            Bucket="test_bucket", Key="users/test_user/doc.pkl"
        )
        storage.redis_connection.delete.assert_called_once_with(
            "users/test_user/doc.pkl"
        )
