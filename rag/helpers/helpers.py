import math

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
