import os
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from constants.constants import SEARCH_LIMIT
from custom_types.custom_types import Document
from helpers.helpers import cosine_similarity, semantic_chunk

model = SentenceTransformer(os.getenv("SENTENCE_TRANSFORMERS_MODEL_NAME"))


# semantic indexing class with chunking
class SemanticIndex:
    # the parts this index contributes to a saved snapshot (see HybridSearch)
    PARTS = ("chunk_embeddings", "chunk_metadata")

    def __init__(
        self,
        docmap: dict[str, Document] | None = None,
        embedding_model: Any = None,
    ) -> None:
        # the process-wide model unless a caller (a test) injects one
        self.model = model if embedding_model is None else embedding_model
        # HybridSearch passes the docmap it shares with the inverted index
        self.docmap: dict[str, Document] = {} if docmap is None else docmap
        # one row per chunk, aligned with chunk_metadata; None until the
        # first chunk is embedded
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict] = []

    # generate an embedding using the model for a text
    def generate_embedding(self, text: str):
        # check if the text is empty
        if text.strip() == "":
            raise ValueError("text to be embedded is empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    # add or replace documents: drop the chunks of every given document id,
    # then embed only these documents' chunks and append them, so the cost
    # scales with the change rather than with the whole corpus
    def build_chunk_embeddings(self, documents: list[Document]):
        # the last copy of a repeated document id wins, as in the docmap
        documents = list({document.id: document for document in documents}.values())
        self._drop_chunks({document.id for document in documents})

        chunks = []
        chunk_metadata = []
        for document in documents:
            # keep the docmap current so chunks can always be hydrated back
            # to their document through the stable document_id
            self.docmap[document.id] = document

            # create chunks from the document contents; blank content has none
            curr_chunks = semantic_chunk(document.content, 4, 1)
            for j, chunk in enumerate(curr_chunks):
                chunks.append(chunk)
                chunk_metadata.append(
                    {
                        "document_id": document.id,
                        "chunk_index": j,
                        "total_chunks": len(curr_chunks),
                    }
                )

        if chunks:
            embeddings = np.asarray(self.model.encode(chunks))
            if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
                self.chunk_embeddings = embeddings
            else:
                self.chunk_embeddings = np.vstack([self.chunk_embeddings, embeddings])
            self.chunk_metadata.extend(chunk_metadata)

        return self.chunk_embeddings

    # remove a document's chunks and its docmap entry; also the delete path
    def remove_document(self, doc_id: str) -> None:
        self._drop_chunks({doc_id})
        self.docmap.pop(doc_id, None)

    # the persisted state; the docmap is saved once, by HybridSearch
    def export_parts(self) -> dict[str, Any]:
        return {
            "chunk_embeddings": self.chunk_embeddings,
            "chunk_metadata": self.chunk_metadata,
        }

    # restore state saved by export_parts; never builds anything
    def restore_parts(self, parts: dict[str, Any]) -> None:
        embeddings = parts["chunk_embeddings"]
        metadata = parts["chunk_metadata"]
        rows = 0 if embeddings is None else len(embeddings)
        if rows != len(metadata):
            raise ValueError(
                f"chunk_embeddings has {rows} rows but chunk_metadata has "
                f"{len(metadata)} entries"
            )
        self.chunk_embeddings = embeddings
        self.chunk_metadata = metadata

    # drop the embedding rows and metadata of the given documents together,
    # so the two stay aligned
    def _drop_chunks(self, doc_ids: set[str]) -> None:
        if self.chunk_embeddings is None:
            return
        keep = [
            i
            for i, metadata in enumerate(self.chunk_metadata)
            if metadata["document_id"] not in doc_ids
        ]
        if len(keep) == len(self.chunk_metadata):
            return
        self.chunk_embeddings = self.chunk_embeddings[keep]
        self.chunk_metadata = [self.chunk_metadata[i] for i in keep]

    # semantic chunk search
    def search_chunks(self, query: str, limit: int = SEARCH_LIMIT):
        # nothing to search: an empty index, or a blank query
        if self.chunk_embeddings is None or not self.chunk_metadata:
            return []
        if not query.strip():
            return []

        # generate an embedding from the query
        query_embedding = self.generate_embedding(query)

        # document similarity_scores
        document_scores: dict[str, tuple[float, dict]] = {}
        # iterate over the chunks
        for i in range(len(self.chunk_embeddings)):
            # create a similarity score between the query embedding and current chunk embedding
            score = cosine_similarity(query_embedding, self.chunk_embeddings[i])
            # get chunk metadata
            metadata = self.chunk_metadata[i]
            doc_id = metadata["document_id"]
            # if the document score does not exist create a new one
            if doc_id not in document_scores or score > document_scores[doc_id][0]:
                document_scores[doc_id] = (score, metadata)

        # get the top documents using the limit
        top_documents = sorted(
            document_scores.items(), key=lambda kv: kv[1][0], reverse=True
        )[:limit]
        # from the top documents create the result that will be sent
        results = []
        for document_id, (score, metadata) in top_documents:
            document = self.docmap.get(document_id)
            if document is None:
                continue
            result = {
                "id": document.id,
                "content": document.content,
                "score": round(score, 4),
                "metadata": metadata,
            }
            results.append(result)

        return results
