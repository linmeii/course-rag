#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Resolve the lib path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.lib.keyword_search import keyword_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = keyword_search(args.query)
            print(f"Searching for: {args.query}")
            for idx, movie in enumerate(results, 1):
                print(f"{idx}. {movie.title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
