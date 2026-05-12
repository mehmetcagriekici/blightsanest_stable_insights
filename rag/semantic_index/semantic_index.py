import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch._prims_common import Tensor
from helpers.helpers import cosine_similarity, semantic_chunk
from types.types import Document

# semantic indexing class with chunking
class SemanticIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = None
        self.docmap = {}
        self.chunk_embeddings = None
        self.chunk_metadata = None

        # for local development
        self.chunk_embeddings_path = Path("./cache/chunk_embeddings.npy")
        self.chunk_metadata_path = Path("./cache/chunk_metadata.json")

    # generate an embedding using the model for a text
    def generate_embedding(self, text: str):
        # check if the text is empty
        if text.strip() == "":
            raise ValueError("text to be embedded is empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    # build embeddings for the documents
    def build_chunk_embeddings(self, documents: list[Document]) -> Tensor:
        self.documents = documents
        # lists to keep chunks and chunk metedata
        chunks = []
        chunk_metadata = []

        # iterate over the documents
        for i in range(len(documents)):
            document = documents[i]
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
                        "document_index": i,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        }
                # add chunk metadata to chunk metadata
                chunk_metadata.append(metadata)

        # create embeddings from the chunks
        self.chunk_embeddings = self.model.encode(chunks)
        # assign chunk metadata
        self.chunk_metadata = chunk_metadata

        # for local development
        Path("./cache").mkdir(exist_ok=True)
        with self.chunk_embeddings_path.open(mode="wb") as f:
            np.save(f, self.chunk_embeddings)
        with self.chunk_metadata_path.open(mode="wb") as f:
            np.save(f, self.chunk_metadata)

        return self.chunk_embeddings

    # load or create chunk embeddings
    def create_or_load_chunk_embeddings(self, documents: list[Document]) -> Tensor:
        self.documents = documents
        # iterate over the documents and create the docmap
        for i in range(len(self.documents)):
            self.docmap[self.documents[i].id] = self.documents[i]
        
        # for local development
        # check if document embeddings are already built
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            # load embeddings
            with self.chunk_embeddings_path.open(mode="rb") as f:
                self.chunk_embeddings = np.load(f)
            # load metadata
            with self.chunk_metadata_path.open(mode="rb") as f:
                self.chunk_metadata = np.load(f)
            # return the chunk embeddings early
            return self.chunk_embeddings
        # otherwise build the embeddings
        return self.build_chunk_embeddings(documents)

    # semantic chunk search
    def search_chunks(self, query: str, limit: int = 10):
        # make sure chunk embeddings exists
        if self.chunk_embeddings is None:
            raise ValueError("chunk embedings is none")

        # make sure chunk metadata exists
        if self.chunk_metadata is None:
            raise ValueError("chunk metadata is none")

        # if the documents do not exist
        if self.documents is None:
            raise ValueError("documents is none")

        # generate an embedding from the query
        query_embedding = self.generate_embedding(query)

        # chunk similarity scores with the query embeddings with metadata
        chunk_scores = []
        # document similarity_scores
        document_scores = {}
        # iterate over the chunks
        for i in range(len(self.chunk_embeddings)):
            # create a similarity score between the query embedding and current chunk embedding
            similarity_score = cosine_similarity(query_embedding, self.chunk_embeddings[i])
            # get chunk metadata
            metadata = self.chunk_metadata[i]
            # create a score struct using the score and the metadata
            chunk_score = {
                    "chunk_index": i,
                    "document_index": metadata["document_index"],
                    "score": similarity_score,
                    }
            chunk_scores.append(chunk_score)
            # if the document score does not exist create a new one
            if metadata["document_index"] not in document_scores:
                document_scores[metadata["document_index"]] = similarity_score
            elif document_scores[metadata["document_index"]] < similarity_score:
                # otherwise if the current score is larger than the previous one update it
                document_scores[metadata["document_index"]] = similarity_score

        # get the top documents using the limit
        top_documents = sorted(document_scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        # from the top documents create the result that will be sent
        results = []
        for kv in top_documents:
            document_index = kv[0]
            document = self.documents[document_index]
            metadata = list(filter(lambda d: d["document_index"] == document_index, self.chunk_metadata))
            # get the first metadata
            if len(metadata) > 0:
                metadata = metadata[0]
            result = {
                    "id": document.id,
                    "content": document.content,
                    "score": round(kv[1], 4),
                    "metadata": metadata or {},
                    }
            results.append(result)

        return results

































