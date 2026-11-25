"""Rank a small corpus with BM25 or evaluate saved retrieval rankings offline."""

import argparse
from collections import Counter, defaultdict
import json
from math import fsum, log1p
from pathlib import Path
import re
import unicodedata


def unique_object(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("JSON object contains duplicate keys")
    return obj


def is_id(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def read_mapping(path):
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or not value or not all(map(is_id, value)):
        raise ValueError("expected a nonempty JSON object with nonblank string IDs")
    return value


def validate_rankings(mapping, *, judgments=False):
    for entries in mapping.values():
        if not isinstance(entries, list) or not all(map(is_id, entries)):
            raise ValueError("expected a list of document IDs for each query")
        if len(set(entries)) != len(entries):
            raise ValueError("a query contains duplicate document IDs")
        if judgments and not entries:
            raise ValueError("each judged query needs at least one relevant document")


def score(truth, rankings, k):
    if truth.keys() != rankings.keys():
        raise ValueError("rankings must contain exactly the judged query IDs")
    rows = []
    for query_id, relevant in sorted(truth.items()):
        relevant = set(relevant)
        hits = [rank for rank, doc_id in enumerate(rankings[query_id][:k], start=1) if doc_id in relevant]
        rows.append({"query_id": query_id, "recall": len(hits) / len(relevant),
                     "reciprocal_rank": 1 / hits[0] if hits else 0, "hit": bool(hits)})
    return {"mean_recall": fsum(row["recall"] for row in rows) / len(rows),
            "mean_reciprocal_rank": fsum(row["reciprocal_rank"] for row in rows) / len(rows),
            "hit_rate": sum(row["hit"] for row in rows) / len(rows), "per_query": rows}


def tokenize(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", text.casefold()))


def rank_bm25(documents, queries, k):
    if any(not isinstance(text, str) or not text.strip() for mapping in (documents, queries) for text in mapping.values()):
        raise ValueError("document and query contents must be nonblank strings")
    postings = defaultdict(list)
    lengths = {}
    for doc_id, text in documents.items():
        counts = Counter(tokenize(text))
        lengths[doc_id] = sum(counts.values())
        for token, count in counts.items():
            postings[token].append((doc_id, count))

    average_length = sum(lengths.values()) / len(lengths) or 1.0
    k1, b = 1.2, 0.75
    rankings = {}
    for query_id, text in sorted(queries.items()):
        scores = defaultdict(float)
        for term in sorted(set(tokenize(text))):
            matches = postings.get(term, ())
            idf = log1p((len(documents) - len(matches) + 0.5) / (len(matches) + 0.5))
            for doc_id, count in matches:
                normalization = k1 * (1 - b + b * lengths[doc_id] / average_length)
                scores[doc_id] += idf * count * (k1 + 1) / (count + normalization)
        rankings[query_id] = sorted(scores, key=lambda doc: (-scores[doc], doc))[:k]
    return rankings


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def recall_fraction(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("expected a fraction between 0 and 1")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    scoring = actions.add_parser("score", help="score saved rankings")
    scoring.add_argument("judgments", type=Path)
    scoring.add_argument("rankings", type=Path)
    scoring.add_argument("--baseline", type=Path)
    scoring.add_argument("--min-recall", type=recall_fraction)
    ranking = actions.add_parser("rank", help="generate rankings with local BM25")
    ranking.add_argument("corpus", type=Path)
    ranking.add_argument("queries", type=Path)
    for action in (ranking, scoring):
        action.add_argument("--k", type=positive_int, default=5)
    args = parser.parse_args(argv)

    try:
        if args.command == "rank":
            report = rank_bm25(read_mapping(args.corpus), read_mapping(args.queries), args.k)
            failed = False
        else:
            truth, rankings = read_mapping(args.judgments), read_mapping(args.rankings)
            validate_rankings(truth, judgments=True)
            validate_rankings(rankings)
            report = {"k": args.k, "queries": len(truth), **score(truth, rankings, args.k)}
            if args.baseline:
                baseline = read_mapping(args.baseline)
                validate_rankings(baseline)
                metrics = score(truth, baseline, args.k)
                report["baseline"] = {key: metrics[key] for key in ("mean_recall", "mean_reciprocal_rank", "hit_rate")}
                report["delta"] = {key: report[key] - value for key, value in report["baseline"].items()}
            failed = args.min_recall is not None and report["mean_recall"] < args.min_recall
    except (OSError, ValueError, RecursionError) as error:
        parser.error(str(error))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
