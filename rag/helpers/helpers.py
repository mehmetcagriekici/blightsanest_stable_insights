import math
import numpy as np
import nltk

from typing import cast

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from torch._prims_common import Tensor

nltk.download('punkt_tab')
nltk.download('stopwords')

# helper function to tokenize a string
def tokenize(text: str) -> list[str]:
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return [w for w in tokens if w.lower() not in stop_words]

# helper function to calc idf score
def calc_idf(total_doc_count: int, term_match_doc_count: int) -> float:
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

# helper function to calc tfidf score
def calc_tf_idf(tf: float, idf: float) -> float:
    return tf * idf

# helper function to calculate cosine similarity
def cosine_similarity(v1: Tensor, v2: Tensor) -> float:
    dot_product: float = np.dot(v1, v2)
    norm1: float = cast(float, np.linalg.norm(v1))
    norm2: float = cast(float, np.linalg.norm(v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# helper function to chunk texts
# base function
def base_chunk(words: list[str], chunk_size: int, overlap: int) -> list[str]:
    # keep track of the chunking
    pivot = 0
    # chunk start
    left_index = 0
    # chunk end
    right_index = chunk_size
    # if chunk size is larger than the words readjust the right_index
    if chunk_size >= len(words):
        right_index = len(words)

    # list to hold chunks
    chunks = []

    while pivot < len(words):
        # if the pivot is at the start of the chunk
        if pivot == left_index:
            pass
    
# improved semantic chunk, built on the base
