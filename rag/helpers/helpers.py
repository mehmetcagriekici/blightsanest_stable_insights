import re
import json
import numpy as np
import nltk

from typing import cast

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')

# helper function to tokenize a string
def tokenize(text: str) -> list[str]:
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return [w for w in tokens if w.lower() not in stop_words]

# helper function to calculate cosine similarity
def cosine_similarity(v1, v2) -> float:
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
        # adjust the right and left index to create the current chunk
        if pivot == left_index:
            # if the start of the chunk is greater than zero and there is overlap
            if left_index > 0 and overlap > 0:
                left_index -= overlap
            # if the current chunk's size is larger than the chunk size
            if right_index - left_index > chunk_size:
                right_index = left_index + chunk_size

            # create the current chunk
            chunk = " ".join(words[left_index:right_index])
            chunk = chunk.strip()

            # if the chunk is empty move to the next iteration
            if chunk == "":
                continue

            # otherwise push it to the chunks
            chunks.append(chunk)

            # assign left index to write index for the next iteration
            left_index = right_index
            # move right index to the end of the next chunk
            right_index += chunk_size
        # move pivot to the next element
        pivot += 1

    return chunks

    
# improved semantic chunk, built on the base
def semantic_chunk(text: str, size: int, overlap: int) -> list[str]:
    # remove gaps
    text = text.strip()
    if text == "":
        return []

    # split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # if there is only one sentence treat it as a single chunk
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]
    return base_chunk(sentences, size, overlap)

# function to calculate rrf score
def calc_rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (rank + k)

# helper function to parse json strings into objects
def parse_json(json_string: str):
    return json.loads(json_string)
























