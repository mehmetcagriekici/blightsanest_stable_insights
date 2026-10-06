from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from custom_types.custom_types import Document
from search.hybrid_search import (
    MANIFEST,
    SNAPSHOT_PARTS,
    CorruptIndexError,
    HybridSearch,
    IndexNotBuiltError,
)


# HybridSearch with stubbed sub-searches: no model, storage, or indexes needed
def make_hybrid(bm25: dict[str, float], semantic: list[dict]) -> HybridSearch:
    search = HybridSearch(Mock(), Mock())
    search.bm25_search = lambda query, limit: bm25
    search.semantic_search = lambda query, limit: semantic
    return search


class TestRrfSearch:
    # the union of both result sets can hold up to 2 x limit documents
    def test_respects_limit(self):
        search = make_hybrid(
            {"a": 3.0, "b": 2.0},
            [
                {"id": "c", "score": 0.9, "content": "c"},
                {"id": "d", "score": 0.8, "content": "d"},
            ],
        )
        assert len(search.rrf_search("q", limit=2)) == 2
        assert len(search.rrf_search("q", limit=10)) == 4

    def test_blank_query_returns_empty(self):
        search = make_hybrid({"a": 1.0}, [{"id": "a", "score": 1.0, "content": "a"}])
        assert search.rrf_search("") == []
        assert search.rrf_search("   ") == []

    # R32 regression: content comes from the one shared docmap
    def test_content_comes_from_the_shared_docmap(self):
        search = make_hybrid({"a": 1.0}, [{"id": "b", "score": 1.0}])
        search.docmap.update(
            {"a": Document(id="a", content="A"), "b": Document(id="b", content="B")}
        )
        contents = {r["doc_id"]: r["content"] for r in search.rrf_search("q")}
        assert contents == {"a": "A", "b": "B"}


DOCUMENTS = [
    Document(id="doc1", content="Today I felt anxious about my presentation."),
    Document(id="doc2", content="I slept well and felt energized."),
    Document(id="doc3", content="Had a productive meeting with the team."),
]


def s3_keys(storage) -> set[str]:
    prefix = f"users/{storage.database_user.id}/"
    listed = storage.s3_client.list_objects_v2(
        Bucket=storage.bucket_name, Prefix=prefix
    )
    return {obj["Key"].removeprefix(prefix) for obj in listed.get("Contents", [])}


def built(storage, embedding_model, documents=DOCUMENTS) -> HybridSearch:
    search = HybridSearch.load_or_empty(storage, embedding_model)
    search.build(documents)
    search.save()
    return search


class TestIndexIsShared:
    def test_both_indexes_use_one_docmap(self):
        search = HybridSearch(Mock(), Mock())
        assert search.inverted_index.docmap is search.docmap
        assert search.semantic_index.docmap is search.docmap

    def test_remove_documents_clears_both_indexes(self, embedding_model):
        search = HybridSearch(Mock(), embedding_model)
        search.build(DOCUMENTS)
        search.remove_documents(["doc1"])
        assert "doc1" not in search.docmap
        assert "doc1" not in search.inverted_index.doc_lengths
        assert all(
            m["document_id"] != "doc1" for m in search.semantic_index.chunk_metadata
        )


# R3 regression: the query path must never build or write an index
class TestQueryPathNeverBuilds:
    def test_load_without_a_saved_index_raises_and_writes_nothing(
        self, s3_storage, embedding_model
    ):
        with pytest.raises(IndexNotBuiltError):
            HybridSearch.load(s3_storage, embedding_model)
        assert s3_keys(s3_storage) == set()

    def test_load_or_empty_without_a_saved_index_is_empty(
        self, s3_storage, embedding_model
    ):
        search = HybridSearch.load_or_empty(s3_storage, embedding_model)
        assert search.manifest is None
        assert search.docmap == {}
        assert s3_keys(s3_storage) == set()


class TestSnapshot:
    def test_save_writes_every_part_then_the_manifest(
        self, s3_storage, embedding_model
    ):
        search = built(s3_storage, embedding_model)
        version = search.manifest["version"]
        assert s3_keys(s3_storage) == {MANIFEST} | {
            f"snapshots/{version}/{part}" for part in SNAPSHOT_PARTS
        }
        assert search.manifest["previous"] is None

    # the manifest is mutable, so it must always be read fresh from s3
    def test_manifest_is_not_cached_but_parts_are(
        self, s3_storage, embedding_model, fake_redis
    ):
        search = built(s3_storage, embedding_model)
        user_prefix = f"users/{s3_storage.database_user.id}/"
        assert f"{user_prefix}{MANIFEST}" not in fake_redis.data
        version = search.manifest["version"]
        assert f"{user_prefix}snapshots/{version}/docmap" in fake_redis.data

    def test_load_restores_the_same_results(self, s3_storage, embedding_model):
        search = built(s3_storage, embedding_model)
        loaded = HybridSearch.load(s3_storage, embedding_model)
        assert loaded.manifest == search.manifest
        assert loaded.docmap == search.docmap
        assert loaded.rrf_search("felt anxious") == search.rrf_search("felt anxious")
        assert loaded.rrf_search("felt anxious")[0]["doc_id"] == "doc1"

    def test_update_and_delete_then_reload(self, s3_storage, embedding_model):
        built(s3_storage, embedding_model)
        search = HybridSearch.load_or_empty(s3_storage, embedding_model)
        search.build([Document(id="doc2", content="A quiet walk in the park.")])
        search.remove_documents(["doc3"])
        search.save()

        loaded = HybridSearch.load(s3_storage, embedding_model)
        assert set(loaded.docmap) == {"doc1", "doc2"}
        assert loaded.rrf_search("park walk")[0]["doc_id"] == "doc2"
        assert "doc3" not in {r["doc_id"] for r in loaded.rrf_search("meeting team")}

    # keep the live snapshot and the one before it (for readers mid-load)
    def test_save_prunes_all_but_the_last_two_snapshots(
        self, s3_storage, embedding_model
    ):
        versions = []
        for _ in range(3):
            search = HybridSearch.load_or_empty(s3_storage, embedding_model)
            search.build(DOCUMENTS)
            search.save()
            versions.append(search.manifest["version"])

        live = {key.split("/")[1] for key in s3_keys(s3_storage) if key != MANIFEST}
        assert live == set(versions[1:])
        assert search.manifest["previous"] == versions[1]

    # R7/R8/R30 regression: a failed save raises, keeps the old snapshot
    # live, and leaves no orphaned parts
    def test_failed_save_raises_and_keeps_the_previous_snapshot(
        self, s3_storage, embedding_model
    ):
        search = built(s3_storage, embedding_model)
        keys_before = s3_keys(s3_storage)

        real_upload = s3_storage.upload_data

        def failing_upload(name, data, cache=True):
            if name.endswith("/chunk_embeddings"):
                raise ClientError({"Error": {"Code": "500"}}, "PutObject")
            real_upload(name, data, cache)

        s3_storage.upload_data = failing_upload
        search.build([Document(id="doc4", content="Never saved.")])
        with pytest.raises(ClientError):
            search.save()

        assert s3_keys(s3_storage) == keys_before
        s3_storage.upload_data = real_upload
        assert "doc4" not in HybridSearch.load(s3_storage, embedding_model).docmap

    def test_missing_part_is_reported_as_corrupt(self, s3_storage, embedding_model):
        search = built(s3_storage, embedding_model)
        s3_storage.delete_data(f"snapshots/{search.manifest['version']}/docmap")
        with pytest.raises(CorruptIndexError, match="docmap"):
            HybridSearch.load(s3_storage, embedding_model)

    def test_misaligned_semantic_parts_are_reported_as_corrupt(
        self, s3_storage, embedding_model
    ):
        search = built(s3_storage, embedding_model)
        version = search.manifest["version"]
        metadata = s3_storage.load_data(f"snapshots/{version}/chunk_metadata")
        s3_storage.upload_data(f"snapshots/{version}/chunk_metadata", metadata[:-1])
        with pytest.raises(CorruptIndexError, match="rows"):
            HybridSearch.load(s3_storage, embedding_model)

    # R29 regression: an s3 outage must surface, not look like "not built"
    def test_s3_outage_raises_instead_of_looking_unbuilt(
        self, s3_storage, embedding_model
    ):
        built(s3_storage, embedding_model)
        s3_storage.s3_client = Mock()
        s3_storage.s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "GetObject"
        )
        with pytest.raises(ClientError):
            HybridSearch.load(s3_storage, embedding_model)
