import logging
from typing import Any, Self
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError

from constants.constants import SEARCH_LIMIT
from custom_types.custom_types import Document
from helpers.helpers import calc_rrf_score
from inverted_index.inverted_index import InvertedIndex
from semantic_index.semantic_index import SemanticIndex
from storage.storage import Storage

logger = logging.getLogger(__name__)

# the only mutable key: names the live snapshot. never cached in redis, so
# every load sees the latest save
MANIFEST = "manifest"
# every part of a snapshot, all stored under snapshots/{version}/
SNAPSHOT_PARTS = ("docmap", *InvertedIndex.PARTS, *SemanticIndex.PARTS)


# the user has no saved index yet: ingestion has to build one first
class IndexNotBuiltError(LookupError):
    pass


# the manifest names a snapshot whose parts are missing or unreadable
class CorruptIndexError(RuntimeError):
    pass


def _part_key(version: str, part: str) -> str:
    return f"snapshots/{version}/{part}"


# blightsanest main search engine: the BM25 and semantic indexes of one user,
# sharing one docmap, saved and loaded together as one versioned snapshot.
#
# query path:     HybridSearch.load(storage), then search. never builds.
# ingestion path: HybridSearch.load_or_empty(storage), then build() and/or
#                 remove_documents(), then save().
#
# save() writes every part under a new snapshots/{version}/ prefix and then
# switches the manifest to it, so a reader sees either the old snapshot or
# the new one, never a mix. snapshot parts are immutable, so caching them in
# redis is always safe. saves for one user must not run concurrently: the
# last manifest write wins and the other save's changes are lost.
class HybridSearch:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        # the one docmap both indexes read and update
        self.docmap: dict[str, Document] = {}
        self.inverted_index = InvertedIndex(self.docmap)
        self.semantic_index = SemanticIndex(self.docmap)
        # the loaded or last saved manifest; None until one exists
        self.manifest: dict[str, Any] | None = None

    # query path: load the saved snapshot, never build one
    @classmethod
    def load(cls, storage: Storage) -> Self:
        search = cls.load_or_empty(storage)
        if search.manifest is None:
            raise IndexNotBuiltError(
                f"no index has been built for user {storage.database_user.id}"
            )
        return search

    # ingestion path: the saved snapshot to update, or an empty index for a
    # user who has none yet
    @classmethod
    def load_or_empty(cls, storage: Storage) -> Self:
        search = cls(storage)
        manifest = storage.load_data(MANIFEST, cache=False)
        if manifest is None:
            return search

        version = manifest["version"]
        parts = {}
        for part in SNAPSHOT_PARTS:
            try:
                parts[part] = storage.load_data(_part_key(version, part))
            except (TypeError, ValueError, KeyError) as e:
                raise CorruptIndexError(
                    f"snapshot {version} part {part} is unreadable"
                ) from e
        missing = [part for part, data in parts.items() if data is None]
        # chunk_embeddings is legitimately None when no document has content
        missing = [part for part in missing if part != "chunk_embeddings"]
        if missing:
            raise CorruptIndexError(f"snapshot {version} is missing {missing}")

        search.docmap.update(parts["docmap"])
        search.inverted_index.restore_parts(parts)
        try:
            search.semantic_index.restore_parts(parts)
        except ValueError as e:
            raise CorruptIndexError(f"snapshot {version}: {e}") from e
        search.manifest = manifest
        return search

    # add or replace documents in both indexes (in memory; call save())
    def build(self, documents: list[Document]) -> None:
        self.inverted_index.build(documents)
        self.semantic_index.build_chunk_embeddings(documents)

    # remove documents from both indexes (in memory; call save())
    def remove_documents(self, doc_ids: list[str]) -> None:
        for doc_id in doc_ids:
            self.inverted_index.remove_document(doc_id)
            self.semantic_index.remove_document(doc_id)

    # write a new snapshot, then point the manifest at it. any failure
    # raises, and the previous snapshot stays live
    def save(self) -> None:
        version = uuid4().hex
        parts = {
            "docmap": self.docmap,
            **self.inverted_index.export_parts(),
            **self.semantic_index.export_parts(),
        }

        written = []
        try:
            for part in SNAPSHOT_PARTS:
                self.storage.upload_data(_part_key(version, part), parts[part])
                written.append(part)
        except Exception:
            # the manifest was never switched; don't leave orphaned parts
            self._delete_parts(version, written)
            raise

        # deliberately outside the cleanup above: a failed response does not
        # prove the manifest write failed, so the new parts may already be
        # live. orphaned parts are the safe failure; deleting live ones is not
        previous = self.manifest
        manifest = {
            "version": version,
            "previous": previous["version"] if previous else None,
        }
        self.storage.upload_data(MANIFEST, manifest, cache=False)
        self.manifest = manifest

        # keep the snapshot just replaced for readers still loading it, and
        # delete the one before it
        if previous and previous.get("previous"):
            self._delete_parts(previous["previous"], SNAPSHOT_PARTS)

    # best-effort cleanup: a failure leaves unused objects behind, which is
    # wasteful but never makes an index wrong
    def _delete_parts(self, version: str, parts) -> None:
        for part in parts:
            try:
                self.storage.delete_data(_part_key(version, part))
            except (ClientError, BotoCoreError) as e:
                logger.error(
                    "could not delete snapshot %s part %s: %s", version, part, e
                )

    # bm25 search from inverted index
    def bm25_search(self, query: str, limit: int = SEARCH_LIMIT):
        return self.inverted_index.bm25_search(query, limit)

    # semantic search from semantic index with chunking
    def semantic_search(self, query: str, limit: int = SEARCH_LIMIT):
        return self.semantic_index.search_chunks(query, limit)

    # rrf search
    def rrf_search(self, query: str, limit: int = SEARCH_LIMIT):
        if not query.strip():
            return []

        # get the bm25 search results
        bm25_results = self.bm25_search(query, limit)
        # sort bm25 results into a list
        bm25_results = sorted(bm25_results.items(), key=lambda kv: kv[1], reverse=True)
        # from bm25 scores create ranks
        bm25_ranks = {}
        for i in range(len(bm25_results)):
            bm25_ranks[bm25_results[i][0]] = i + 1

        # get semantic search results
        semantic_results = self.semantic_search(query, limit)
        # sort semantic results
        semantic_results = sorted(
            semantic_results, key=lambda score: score["score"], reverse=True
        )
        # from semantic scores create ranks
        semantic_ranks = {}
        for i in range(len(semantic_results)):
            semantic_ranks[semantic_results[i]["id"]] = i + 1

        # fuse over the union of both result sets so a document ranked highly
        # by only one method is not silently dropped
        doc_ids = set(bm25_ranks) | set(semantic_ranks)

        # calculate rrf scores
        rrf_scores = []
        for doc_id in doc_ids:
            rrf_score = 0.0
            semantic_rank = 0
            bm25_rank = 0

            if doc_id in semantic_ranks:
                semantic_rank = semantic_ranks[doc_id]
                rrf_score += calc_rrf_score(semantic_rank)
            if doc_id in bm25_ranks:
                bm25_rank = bm25_ranks[doc_id]
                rrf_score += calc_rrf_score(bm25_rank)

            # both indexes share one docmap, so content has a single source
            document = self.docmap.get(doc_id)
            content = document.content if document is not None else ""

            # create the rrf score object and append it to the scores
            rrf_scores.append(
                {
                    "doc_id": doc_id,
                    "content": content,
                    "bm25_rank": bm25_rank,
                    "semantic_rank": semantic_rank,
                    "rrf_score": rrf_score,
                }
            )

        # sort the rrf scores
        return sorted(rrf_scores, key=lambda score: score["rrf_score"], reverse=True)[
            :limit
        ]
