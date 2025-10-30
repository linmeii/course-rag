from dataclasses import dataclass
from pathlib import Path
import json

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_DIR / "data"

MOVIES_PATH = DATA_PATH / "movies.json"
STOPWORDS_PATH = DATA_PATH / "stopwords.txt"


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
