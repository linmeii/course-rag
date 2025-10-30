from collections import Counter
import math
from .search_utils import ROOT_DIR, Movie, load_movies, tokenize_text
from pickle import dump, load

CACHE_DIR = ROOT_DIR / "cache"
INDEX_CACHE_PATH = CACHE_DIR / "index.pkl"
DOCMAP_CACHE_PATH = CACHE_DIR / "docmap.pkl"
TERM_FREQUENCIES_CACHE_PATH = CACHE_DIR / "term_frequencies.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies.setdefault(doc_id, Counter())
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.term_frequencies[doc_id][token] += 1
            self.index[token].add(doc_id)

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

    def load(self) -> None:
        if not INDEX_CACHE_PATH.exists():
            raise FileNotFoundError("Index cache file not found")
        if not DOCMAP_CACHE_PATH.exists():
            raise FileNotFoundError("Docmap cache file not found")
        if not TERM_FREQUENCIES_CACHE_PATH.exists():
            raise FileNotFoundError("Term frequencies cache file not found")
        with open(INDEX_CACHE_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_CACHE_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TERM_FREQUENCIES_CACHE_PATH, "rb") as f:
            self.term_frequencies = load(f)
