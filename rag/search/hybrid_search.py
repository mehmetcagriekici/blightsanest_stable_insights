from constants.constants import SEARCH_LIMIT
from helpers.helpers import calc_rrf_score
from semantic_index.semantic_index import SemanticIndex
from inverted_index.inverted_index import InvertedIndex
from custom_types.custom_types import Document, User

# blightsanest main search engine
class HybridSearch:
    def __init__(self, current_user: User, documents: list[Document]) -> None:
        # documents search will run on
        self.documents = documents
        # semantic index
        self.semantic_index = SemanticIndex(current_user)
        # inverted_index
        self.inverted_index = InvertedIndex(current_user)
        
        # local development
        # load the semantic_index
        self.semantic_index.create_or_load_chunk_embeddings(documents)
        # load the inverted index
        self.inverted_index.load(documents)

    # bm25 search from inverted index
    def bm25_search(self, query: str, limit: int = SEARCH_LIMIT):
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
        # from semantic scores create ranks and keep a lookup for content/metadata
        semantic_ranks = {}
        semantic_by_id = {}
        for i in range(len(semantic_results)):
            semantic_ranks[semantic_results[i]["id"]] = i + 1
            semantic_by_id[semantic_results[i]["id"]] = semantic_results[i]

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

            # resolve content from whichever source has it (semantic results
            # carry content; bm25-only docs come from the docmap)
            if doc_id in semantic_by_id:
                content = semantic_by_id[doc_id]["content"]
            else:
                document = self.inverted_index.docmap.get(doc_id)
                content = document.content if document is not None else ""

            # create the rrf score object and append it to the scores
            rrf_scores.append({
                "doc_id": doc_id,
                "content": content,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "rrf_score": rrf_score,
                })

        # sort the rrf scores
        return sorted(rrf_scores, key=lambda score: score["rrf_score"], reverse=True)































