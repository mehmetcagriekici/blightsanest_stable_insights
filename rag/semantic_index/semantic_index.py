import os
from annotated_types import doc
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from helpers.helpers import cosine_similarity
from types.types import Document

class SemanticIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # define MLM model
        self.model = SentenceTransformer(model_name)
        # embeddings will be created by the MLM (model.encode) initiated at None
        self.embeddings = None
        # documents initially none, later list of Document
        self.documents = None
        # a dictionary mapping document ids to their full document objects
        self.docmap = {}

        # for local development
        self.embeddings_path = Path("cache/embeddings.npy")

    # generate embeddings using the model for document content
    def generate_embeddings(self, text: str):
        # check if the content is empty
        if text.strip() == "":
            raise ValueError("content cannot be empty")
        embeddings = self.model.encode([text])
        return embeddings[0]

    # build embeddings for the documents
    def build_embeddings(self, documents: list[Document]):
        self.documents = documents
        # list for document contents to be embedded
        contents = []
        for document in self.documents:
            # add document to the docmap
            self.docmap[document.id] = document
            contents.append(document.content)
        # emded the document contents
        self.embeddings = self.model.encode(contents, show_progress_bar=True)
        # for local development
        Path("./cache").mkdir(exist_ok=True)
        with self.embeddings_path.open(mode="wb") as f:
            np.save(f, self.embeddings)

    # load or create embeddings from the documents
    def load_or_create_embeddings(self, documents: list[Document]):
        # assisgn documents
        self.documents = documents

        # for local development
        # check if embeddings exist in AWS
        if os.path.exists(self.embeddings_path):
            # create the docmap
            for document in self.documents:
                self.docmap[document.id] = document
                with self.embeddings_path.open(mode="rb") as f:
                    self.embeddings = np.load(f)
                if len(self.embeddings) == len(self.documents):
                    return self.embeddings
        return self.build_embeddings(documents)

    # semantic search
    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings exist for search to happen")
        if self.documents is None:
            raise ValueError("No documents exist for search to happen")
        # generate query embeddings
        query_embeddings = self.generate_embeddings(query)
        # find similarities between the query embeddings and document embeddings
        similarities = []
        for i in range(self.embeddings.shape[0]):
            score = cosine_similarity(query_embeddings, self.embeddings[i])
            similarities.append((score, self.documents[i]))
        # sort the similarities and return the result
        return sorted(similarities, key=lambda el: el[0], reverse=True)[:limit]

# main semantic class
class ChunkedSemanticIndex(SemanticIndex):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        # for local development
        self.chunk_embeddings_path = Path("./cache/chunk_embeddings.npy")
        self.chunk_metadata_path = Path("./cache/chunk_metadata.npy")
