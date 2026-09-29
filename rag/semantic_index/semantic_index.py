import logging
import os

from botocore.exceptions import ClientError
from redis import ResponseError
from sentence_transformers import SentenceTransformer

from constants.constants import SEARCH_LIMIT
from custom_types.custom_types import Document
from helpers.helpers import cosine_similarity, semantic_chunk
from storage.storage import Storage

logger = logging.getLogger(__name__)
model = SentenceTransformer(os.getenv("SENTENCE_TRANSFORMERS_MODEL_NAME"))


# semantic indexing class with chunking
class SemanticIndex:
    def __init__(self, storage: Storage) -> None:
        self.model = model
        self.documents = None
        self.docmap = {}
        self.chunk_embeddings = None
        self.chunk_metadata = None

        # storage for embeddings and metadata
        self.storage = storage

    # generate an embedding using the model for a text
    def generate_embedding(self, text: str):
        # check if the text is empty
        if text.strip() == "":
            raise ValueError("text to be embedded is empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    # build embeddings for the documents
    def build_chunk_embeddings(self, documents: list[Document]):
        self.documents = documents
        # lists to keep chunks and chunk metedata
        chunks = []
        chunk_metadata = []

        # iterate over the documents
        for i in range(len(documents)):
            document = documents[i]
            # keep the docmap current so chunks can always be hydrated back
            # to their document through the stable document_id, regardless
            # of whether build_chunk_embeddings is called directly (ingestion)
            # or via create_or_load_chunk_embeddings
            self.docmap[document.id] = document
            # if document content is empty move to the next iteration
            if document.content == "":
                continue

            # create chunks from the document contents
            curr_chunks = semantic_chunk(document.content, 4, 1)
            # iterate over the chunks
            for j in range(len(curr_chunks)):
                # add curr_chunk to the chunks
                chunks.append(curr_chunks[j])
                # create chunk metada
                metadata = {
                    "document_id": document.id,
                    "chunk_index": j,
                    "total_chunks": len(curr_chunks),
                }
                # add chunk metadata to chunk metadata
                chunk_metadata.append(metadata)

        # create embeddings from the chunks
        self.chunk_embeddings = self.model.encode(chunks)
        # assign chunk metadata
        self.chunk_metadata = chunk_metadata

        # upload chunk embedings and chunk metadata to the storage
        try:
            # chunk embeddings
            self.storage.upload_data("chunk_embeddings", self.chunk_embeddings)
            # metadata
            self.storage.upload_data("chunk_metadata", self.chunk_metadata)
        except ValueError as e:
            logger.error(
                "a value error occured while trying to upload the semantic index: %s", e
            )
            return None
        except ClientError as e:
            logger.error(
                "a client error occured while trying to upload the semantic index: %s",
                e,
            )
            return None
        except ResponseError as e:
            logger.error(
                "a response error occured while trying to upload the semantic index: %s",
                e,
            )
            return None

        return self.chunk_embeddings

    # load or create chunk embeddings
    def create_or_load_chunk_embeddings(self, documents: list[Document]):
        self.documents = documents
        # iterate over the documents and create the docmap
        for i in range(len(self.documents)):
            self.docmap[self.documents[i].id] = self.documents[i]

        # check if chunk embeddings and chunk metadata is already built
        chunk_embeddings = self.storage.load_data("chunk_embeddings")
        chunk_metadata = self.storage.load_data("chunk_metadata")
        # use explicit None checks: chunk_embeddings is a numpy array and
        # evaluating it in a boolean context raises ValueError
        if chunk_metadata is not None and chunk_embeddings is not None:
            self.chunk_embeddings = chunk_embeddings
            self.chunk_metadata = chunk_metadata
            return self.chunk_embeddings

        # otherwise build the embeddings
        return self.build_chunk_embeddings(documents)

    # semantic chunk search
    def search_chunks(self, query: str, limit: int = SEARCH_LIMIT):
        # make sure chunk embeddings exists
        if self.chunk_embeddings is None:
            raise ValueError("chunk embedings is none")

        # make sure chunk metadata exists
        if self.chunk_metadata is None:
            raise ValueError("chunk metadata is none")

        # if the documents do not exist
        if self.documents is None:
            raise ValueError("documents is none")

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
