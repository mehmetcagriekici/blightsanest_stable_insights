from semantic_index.semantic_index import SemanticIndex
from inverted_index.inverted_index import InvertedIndex
from types.types import Document

# blightsanest main search engine
class HybridSearch:
    def __init__(self, documents: list[Document]) -> None:
        # documents search will run on
        self.documents = documents
        # semantic index
        self.semantic_index = SemanticIndex()
        # inverted_index
        self.inverted_index = InvertedIndex()
        
        # local development
        # load the semantic_index
        self.semantic_index.create_or_load_chunk_embeddings(documents)
        # load the inverted index
        self.inverted_index.create_or_load_inverted_index(documents)

    # bm25 search from inverted index
    def bm25_search(self, query: str, limit: int = 10):
        # local development only
        # load the index
        self.inverted_index.load()
        return self.inverted_index.bm25_search(query, limit)

    # semantic search from semantic index with chunking
    def semantic_search(self, query: str, limit: int = 10):
        return self.semantic_index.search_chunks(query, limit)

    # rrf search
    def rrf_search(self, query: str, limit: int = 50):
        pass
