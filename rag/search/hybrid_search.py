from constants.constants import SEARCH_LIMIT
from helpers.helpers import calc_rrf_score
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
    def bm25_search(self, query: str, limit: int = SEARCH_LIMIT):
        # local development only
        # load the index
        self.inverted_index.load()
        return self.inverted_index.bm25_search(query, limit)

    # semantic search from semantic index with chunking
    def semantic_search(self, query: str, limit: int = SEARCH_LIMIT):
        return self.semantic_index.search_chunks(query, limit)

    # rrf search
    def rrf_search(self, query: str, limit: int = SEARCH_LIMIT):
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
        semantic_results = sorted(semantic_results, key=lambda score: score["score"], reverse=True)

        # calculate rrf scores
        rrf_scores = []
        for i in range(len(semantic_results)):
            rrf_score = calc_rrf_score(i + 1)
            bm25_rank = 0
            if semantic_results[i]["id"] in bm25_ranks:
                rrf_score += calc_rrf_score(bm25_ranks[semantic_results[i]["id"]])
                bm25_rank = bm25_ranks[semantic_results[i]["id"]]

            # create the rrf score object and append it to the scores
            rrf_scores.append({
                "doc_id": semantic_results[i]["id"],
                "content": semantic_results[i]["content"],
                "bm25_rank": bm25_rank,
                "semantic_rank": i + 1,
                "rrf_score": rrf_score,
                })
        # sort the rrf scores
        return sorted(rrf_scores, key=lambda score: score["rrf_score"], reverse=True)































