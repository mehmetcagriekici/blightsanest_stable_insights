import itertools
import math
import pickle
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path

from constants.constants import BM25_B, BM25_K1
from helpers.helpers import tokenize
from types.types import Document

class InvertedIndex:
    def __init__(self) -> None:
        # a dcitionary mapping tokens to set of document ids
        self.index: dict[str, set[str]] = {}
        # a dictionary mapping document ids to their full document objects
        self.docmap = {}
        # a dictonary mapping document ids to term frequencies
        self.term_frequencies = defaultdict(Counter)
        # a dictionary mapping document ids to their lengths
        self.doc_lengths: dict[str, int] = {}

        # for local and development - will be moved to AWS
        self.index_path = Path("cache/index.pkl")
        self.docmap_path = Path("cache/docmap.pkl")
        self.term_frequencies_path = Path("cache/term_frequencies.pkl")
        self.doc_lengths_path = Path("cache/doc_lengths.pkl") 

    # tokenize document content (text), add each token to the index with the document id
    def add_document(self, text: str, doc_id: str) -> None:
        # tokenize the document content
        tokens = tokenize(text)
        # update the term frequencies of the document
        self.term_frequencies[doc_id].update(tokens)
        # save document length
        self.doc_lengths[doc_id] = len(tokens)

        # fill the index with the tokens
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    # calculate the average doc length across all documents
    def get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        sum = 0
        for doc_id in self.doc_lengths:
            sum += self.doc_lengths[doc_id]
        return sum / len(self.doc_lengths)

    # get the set of document ids of a token
    def get_documents(self, token: str) -> set[str]:
        return self.index.get(token) or set()
    
    # iterate over all the documents and add them to the docmap and the index
    def build(self, documents: list[Document]):
        for doc in documents:
            self.docmap[doc.id] = doc
            self.add_document(doc.content, doc.id)

    # save index, term frequencies, document lengths, and docmap to disk for local development
    # save to AWS
    def save(self):
        Path("./cache").mkdir(exist_ok=True)
        with self.index_path.open(mode="wb") as f:
            pickle.dump(self.index, f)
        with self.docmap_path.open(mode="wb") as f:
            pickle.dump(self.docmap, f)
        with self.term_frequencies_path.open(mode="wb") as f:
            pickle.dump(self.term_frequencies, f)
        with self.doc_lengths_path.open(mode="wb") as f:
            pickle.dump(self.doc_lengths, f)

    # load index, term frequencies, document length, and ocmağ from the disk for local development
    # load from the AWS
    def load(self):
        try:
            with self.index_path.open(mode="rb") as f:
                self.index = pickle.load(f)
            with self.docmap_path.open(mode="rb") as f:
                self.docmap = pickle.load(f)
            with self.term_frequencies_path.open(mode="rb") as f:
                self.term_frequencies = pickle.load(f)
            with self.doc_lengths_path.open(mode="rb") as f:
                self.doc_lengths = pickle.load(f)
        except ValueError as err:
            print(err)
        except OSError as err:
            print(err)

    # get the frequency of a single token
    def get_tf(self, doc_id: str, token: str) -> int:
        # check if the document exists in the term frequencies
        if doc_id not in self.term_frequencies:
            return 0
        return self.term_frequencies[doc_id][token]

    # calculate the idf score of a single term
    def get_idf(self, term: str) -> float:
        # number of documents
        n = len(self.docmap)
        # get document frequency for the given term
        df = len(self.get_documents(term))
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    # calculate the saturated idf score
    def get_bm25_tf(self, doc_id: str, token: str) -> float:
        avg_len = self.get_avg_doc_length()
        length_norm = 1
        # calc length norm of the document
        if avg_len != 0:
            doc_len = self.doc_lengths[doc_id]
            length_norm = 1 - BM25_B + BM25_B * (doc_len / avg_len)
        # get the term frequency of the term
        tf = self.get_tf(doc_id, token)
        return (tf * (BM25_K1 + 1)) / (tf + BM25_K1 * length_norm)

    # calculate bm25 score of a token
    def bm25(self, doc_id: str, token: str) -> float:
        idf = self.get_idf(token)
        tf = self.get_bm25_tf(doc_id, token)
        return idf * tf

    # implement the bm25 search algorithm
    def bm25_search(self, query: str, limit: int=5):
        # tokenize the query
        tokens = tokenize(query)
        scores = defaultdict(float)

        for token in tokens:
            if token in self.index:
                for doc_id in self.index[token]:
                    scores[doc_id] += self.bm25(doc_id, token)
        return OrderedDict(itertools.islice(sorted(scores.items(), key=lambda kv: kv[1], reverse=True), limit))
















