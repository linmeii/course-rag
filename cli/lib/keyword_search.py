from .search_utils import Movie, load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer


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


def keyword_search(query: str) -> list[Movie]:
    results = []
    movies = load_movies()

    query_tokens = tokenize_text(query)
    for movie in movies:
        title_tokens = tokenize_text(movie.title)
        if any(
            query_token in title_token
            for query_token in query_tokens
            for title_token in title_tokens
        ):
            results.append(movie)

    return results[:5]
