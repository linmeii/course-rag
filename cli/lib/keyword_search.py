from .inverted_index import InvertedIndex
from .search_utils import Movie, tokenize_text


def keyword_search(query: str) -> list[Movie]:
    try:
        inverted_index = InvertedIndex()
        inverted_index.load()
    except FileNotFoundError:
        print("Index file not found. Please build the index first.")
        exit(1)

    results: list[Movie] = []
    movie_ids: list[int] = []
    query_tokens = tokenize_text(query)

    for query_token in query_tokens:
        movie_ids.extend(inverted_index.get_documents(query_token))
        if len(movie_ids) == 5:
            break

    results.extend(inverted_index.docmap[movie_id] for movie_id in movie_ids)
    return results
