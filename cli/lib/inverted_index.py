from collections import Counter
import math
from .search_utils import BM25_B, BM25_K1, ROOT_DIR, Movie, load_movies, tokenize_text
from pickle import dump, load

CACHE_DIR = ROOT_DIR / "cache"
INDEX_CACHE_PATH = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE_PATH = CACHE_DIR / "term_frequencies.pkl"
DOC_LENGTHS_CACHE_PATH = CACHE_DIR / "doc_lengths.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term.lower(), set())))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) != 1:
            raise ValueError("get_tf expects a single token, got multiple")
        token = tokenized_term[0]
        return self.term_frequencies[doc_id].get(token, 0)

    def get_idf(self, term: str) -> float:
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) != 1:
            raise ValueError("get_idf expects a single token, got multiple")
        token = tokenized_term[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token, set()))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = tokenize_text(term)
        if len(tokenized_term) != 1:
            raise ValueError("get_bm25_idf expects a single token, got multiple")
        token = tokenized_term[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token, set()))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[Movie, float]:
        tokenized_query = tokenize_text(query)
        bm25_scores: dict[int, float] = {}
        for doc_id in self.docmap:
            doc_total = 0.0
            for query_token in tokenized_query:
                doc_total += self.bm25(doc_id, query_token)
            bm25_scores[doc_id] = doc_total
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.docmap[doc_id], score) for doc_id, score in sorted_scores[:limit]]

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie.id, f"{movie.title} {movie.description}")
            self.docmap[movie.id] = movie

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(INDEX_CACHE_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_CACHE_PATH, "wb") as f:
            dump(self.docmap, f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "wb") as f:
            dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_CACHE_PATH, "wb") as f:
            dump(self.doc_lengths, f)

    def load(self) -> None:
        if not INDEX_CACHE_PATH.exists():
            raise FileNotFoundError("Index cache file not found")
        if not DOCMAP_CACHE_PATH.exists():
            raise FileNotFoundError("Docmap cache file not found")
        if not TERM_FREQUENCIES_CACHE_PATH.exists():
            raise FileNotFoundError("Term frequencies cache file not found")
        if not DOC_LENGTHS_CACHE_PATH.exists():
            raise FileNotFoundError("Doc lengths cache file not found")
        with open(INDEX_CACHE_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_CACHE_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "rb") as f:
            self.term_frequencies = load(f)
        with open(DOC_LENGTHS_CACHE_PATH, "rb") as f:
            self.doc_lengths = load(f)
