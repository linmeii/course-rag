#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Resolve the lib path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.inverted_index import InvertedIndex
from cli.lib.keyword_search import keyword_search
from cli.lib.search_utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency of a term in a document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency of a term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get frequency for")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score of a term in a document"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            results = keyword_search(args.query)
            print(f"Searching for: {args.query}")
            for idx, movie in enumerate(results, 1):
                print(f"{idx}. {movie.title}")
        case "bm25search":
            inverted_index = InvertedIndex()
            inverted_index.load()
            results = inverted_index.bm25_search(args.query, args.limit)
            for idx, (movie, score) in enumerate(results, 1):
                print(f"{idx}. {movie.title} - Score: {score:.2f}")
        case "tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")
        case "idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            idf = inverted_index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            tfidf = inverted_index.get_tfidf(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document {args.doc_id}: {tfidf:.2f}"
            )
        case "bm25idf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            bm25idf = inverted_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            inverted_index = InvertedIndex()
            inverted_index.load()
            bm25tf = inverted_index.get_bm25_tf(args.doc_id, args.term, args.k1)
            print(
                f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}"
            )
        case "build":
            ii = InvertedIndex()
            ii.build()
            ii.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
