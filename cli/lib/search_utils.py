from dataclasses import dataclass
from pathlib import Path
import json
import string
from nltk.stem import PorterStemmer

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_DIR / "data"

MOVIES_PATH = DATA_PATH / "movies.json"
STOPWORDS_PATH = DATA_PATH / "stopwords.txt"

BM25_K1 = 1.5
BM25_B = 0.75


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    description: str


def load_movies() -> list[Movie]:
    with open(MOVIES_PATH, "r") as f:
        data = json.load(f)
        movies = [Movie(**movie) for movie in data["movies"]]
    return movies


def load_stopwords() -> set[str]:
    with open(STOPWORDS_PATH, "r") as f:
        stopwords = set(f.read().splitlines())
    return stopwords


def preprocess_text(text: str) -> str:
    ptext = text.lower().strip()
    ptext = ptext.translate(str.maketrans("", "", string.punctuation))
    return ptext


def tokenize_text(text: str) -> list[str]:
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    return [
        stemmer.stem(text)
        for text in preprocess_text(text).split()
        if text != "" and text not in stopwords
    ]
