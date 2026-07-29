#!/usr/bin/env python3
"""Build ChunkLab's reproducible SQuAD retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQUAD_PATH = DATA / "squad-dev-v1.1.json"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SEED = 20260729
ARTICLE_COUNT = 40
QUESTION_COUNT = 800
KS = (1, 3, 5, 10)
BM25_K1 = 1.2
BM25_B = 0.75

STRATEGIES = (
    {"id": "fixed-500", "name": "Fixed 500", "description": "500 characters, no overlap"},
    {"id": "fixed-10", "name": "Fixed 500 + 10%", "description": "500 characters, 50-character overlap"},
    {"id": "fixed-25", "name": "Fixed 500 + 25%", "description": "500 characters, 125-character overlap"},
    {"id": "recursive", "name": "Recursive", "description": "Paragraph, sentence, whitespace, then hard cut"},
    {"id": "sentence-window", "name": "Sentence window", "description": "Index one sentence, return it with two neighbors"},
    {"id": "parent-document", "name": "Parent document", "description": "Index 250-character children, return 1,000-character parents"},
    {"id": "semantic", "name": "Semantic (TF-IDF proxy)", "description": "Split at the 80th percentile of adjacent-sentence TF-IDF cosine distance"},
    {"id": "fixed-1000", "name": "Fixed 1000", "description": "1,000 characters, no overlap"},
)


def chunk(start: int, end: int, text: str, index_start: int | None = None, index_end: int | None = None) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text[start:end],
        "index_start": start if index_start is None else index_start,
        "index_end": end if index_end is None else index_end,
        "index_text": text[start if index_start is None else index_start : end if index_end is None else index_end],
    }


def fixed_chunks(text: str, size: int, overlap: int = 0) -> list[dict]:
    if not text:
        return []
    step = size - overlap
    return [chunk(start, min(len(text), start + size), text) for start in range(0, len(text), step)]


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    ranges = [(m.start(), m.end()) for m in re.finditer(r"\s*[^.!?]+(?:[.!?]+(?=\s|$)|$)", text)]
    if not ranges:
        return [(0, len(text))] if text else []
    if ranges[-1][1] < len(text):
        ranges.append((ranges[-1][1], len(text)))
    return ranges


def structural_chunks(text: str, size: int = 500) -> list[dict]:
    """Contiguous recursive-like split that keeps separators and every character."""
    if len(text) <= size:
        return [chunk(0, len(text), text)] if text else []
    out, start = [], 0
    separators = ("\n\n", ". ", "? ", "! ", "\n", " ")
    while start < len(text):
        hard_end = min(len(text), start + size)
        if hard_end == len(text):
            end = hard_end
        else:
            floor = start + size // 2
            candidates = []
            for separator in separators:
                pos = text.rfind(separator, floor, hard_end)
                if pos >= 0:
                    candidates.append(pos + len(separator))
            end = max(candidates, default=hard_end)
        out.append(chunk(start, end, text))
        start = end
    return out


def sentence_window_chunks(text: str, neighbors: int = 2) -> list[dict]:
    sentences = sentence_ranges(text)
    return [
        chunk(sentences[max(0, i - neighbors)][0], sentences[min(len(sentences) - 1, i + neighbors)][1], text, start, end)
        for i, (start, end) in enumerate(sentences)
    ]


def parent_document_chunks(text: str, child_size: int = 250, parent_size: int = 1000) -> list[dict]:
    out = []
    for parent_start in range(0, len(text), parent_size):
        parent_end = min(len(text), parent_start + parent_size)
        for child_start in range(parent_start, parent_end, child_size):
            out.append(chunk(parent_start, parent_end, text, child_start, min(parent_end, child_start + child_size)))
    return out


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())


def tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    counts = [Counter(tokenize(text)) for text in texts]
    document_frequency = Counter(term for row in counts for term in row)
    total = len(texts)
    return [
        {
            term: frequency * (math.log((1 + total) / (1 + document_frequency[term])) + 1)
            for term, frequency in row.items()
        }
        for row in counts
    ]


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    denom = math.sqrt(sum(x * x for x in a.values())) * math.sqrt(sum(x * x for x in b.values()))
    return sum(value * b.get(term, 0.0) for term, value in a.items()) / denom if denom else 0.0


def bm25_ranker(texts: list[str], k1: float = BM25_K1, b: float = BM25_B):
    rows = [Counter(tokenize(text)) for text in texts]
    lengths = [sum(row.values()) for row in rows]
    average_length = statistics.fmean(lengths) if lengths else 1.0
    postings = defaultdict(list)
    for index, row in enumerate(rows):
        for term, frequency in row.items():
            postings[term].append((index, frequency))

    def rank(query: str) -> list[int]:
        scores = defaultdict(float)
        for term in dict.fromkeys(tokenize(query)):
            matches = postings.get(term, ())
            idf = math.log1p((len(rows) - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, frequency in matches:
                normalizer = frequency + k1 * (1 - b + b * lengths[index] / average_length)
                scores[index] += idf * frequency * (k1 + 1) / normalizer
        ordered = sorted(scores, key=lambda index: (-scores[index], index))
        ordered.extend(index for index in range(len(rows)) if index not in scores)
        return ordered

    return rank


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def semantic_chunks(text: str, sentence_vectors: list[dict[str, float]], threshold_q: float = 0.8, max_size: int = 500) -> list[dict]:
    sentences = sentence_ranges(text)
    if len(sentences) <= 1:
        return fixed_chunks(text, max_size)
    distances = [1 - cosine(sentence_vectors[i], sentence_vectors[i + 1]) for i in range(len(sentences) - 1)]
    threshold = percentile(distances, threshold_q)
    break_after = {i for i, distance in enumerate(distances) if distance >= threshold}
    out, group_start = [], sentences[0][0]
    for i, (_, sentence_end) in enumerate(sentences):
        must_split = i in break_after or i == len(sentences) - 1 or sentence_end - group_start >= max_size
        if not must_split:
            continue
        while sentence_end - group_start > max_size:
            out.append(chunk(group_start, group_start + max_size, text))
            group_start += max_size
        if sentence_end > group_start:
            out.append(chunk(group_start, sentence_end, text))
        group_start = sentence_end
    return out


def reciprocal_rank(relevances: list[int], k: int = 10) -> float:
    return next((1 / rank for rank, rel in enumerate(relevances[:k], 1) if rel), 0.0)


def ndcg(relevances: list[int], total_relevant: int, k: int = 10) -> float:
    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate_metrics(rankings: list[dict]) -> dict:
    recalls = {k: statistics.fmean(any(row["relevances"][:k]) for row in rankings) for k in KS}
    metrics = {f"recall_at_{k}": recalls[k] for k in KS}
    metrics["mrr_at_10"] = statistics.fmean(reciprocal_rank(row["relevances"]) for row in rankings)
    metrics["ndcg_at_10"] = statistics.fmean(
        ndcg(row["relevances"], row["total_relevant"]) for row in rankings
    )
    threshold_k = next((k for k in KS if recalls[k] > 0.9), None)
    metrics["context_k_to_90"] = threshold_k
    metrics["context_tokens_to_90"] = (
        statistics.fmean(sum(row["context_tokens"][:threshold_k]) for row in rankings)
        if threshold_k
        else None
    )
    return metrics


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def metric_selftest() -> None:
    rows = [
        {"relevances": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], "total_relevant": 1, "context_tokens": [1] * 10},
        {"relevances": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], "total_relevant": 1, "context_tokens": [1] * 10},
    ]
    metrics = aggregate_metrics(rows)
    recalls = [metrics[f"recall_at_{k}"] for k in KS]
    assert recalls == sorted(recalls), "recall@k must be monotone"
    assert reciprocal_rank([0, 0, 1]) == 1 / 3, "MRR first hit at r must equal 1/r"
    assert ndcg([1, 1, 0], 2) == 1.0, "ideal nDCG must equal 1"
    assert bm25_ranker(["alpha alpha", "beta"])("alpha")[0] == 0, "BM25 must rank a matching chunk first"


def load_squad() -> dict:
    DATA.mkdir(exist_ok=True)
    if not SQUAD_PATH.exists():
        print("downloading SQuAD v1.1 dev")
        urllib.request.urlretrieve(SQUAD_URL, SQUAD_PATH)
    return json.loads(SQUAD_PATH.read_text())


def sample_corpus(raw: dict) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    articles = list(raw["data"])
    rng.shuffle(articles)
    documents, questions = [], []
    for article in articles[:ARTICLE_COUNT]:
        parts, article_questions = [], []
        for paragraph in article["paragraphs"]:
            if parts:
                parts.append("\n\n")
            context = paragraph["context"]
            parts.append(context)
            for qa in paragraph["qas"]:
                answers = list(dict.fromkeys(answer["text"] for answer in qa["answers"]))
                article_questions.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answers": answers,
                        "document_title": article["title"],
                    }
                )
        documents.append({"title": article["title"], "text": "".join(parts)})
        questions.extend(article_questions)
    rng.shuffle(questions)
    return documents, questions[:QUESTION_COUNT]


def build_chunks(strategy_id: str, document: dict) -> list[dict]:
    text = document["text"]
    if strategy_id == "fixed-500":
        return fixed_chunks(text, 500)
    if strategy_id == "fixed-10":
        return fixed_chunks(text, 500, 50)
    if strategy_id == "fixed-25":
        return fixed_chunks(text, 500, 125)
    if strategy_id == "recursive":
        return structural_chunks(text)
    if strategy_id == "sentence-window":
        return sentence_window_chunks(text)
    if strategy_id == "parent-document":
        return parent_document_chunks(text)
    if strategy_id == "semantic":
        sentences = sentence_ranges(text)
        return semantic_chunks(text, tfidf_vectors([text[start:end] for start, end in sentences]))
    if strategy_id == "fixed-1000":
        return fixed_chunks(text, 1000)
    raise ValueError(strategy_id)


def token_counter():
    try:
        import tiktoken
    except ImportError:
        return lambda text: math.ceil(len(text.encode()) / 4), True, "estimated UTF-8 bytes / 4"
    encoder = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoder.encode(text)), False, "tiktoken cl100k_base"


def score_strategy(strategy: dict, documents: list[dict], questions: list[dict], count_tokens, token_counts_estimated: bool):
    chunks = []
    chunks_per_doc = []
    for document in documents:
        document_chunks = build_chunks(strategy["id"], document)
        chunks_per_doc.append(len(document_chunks))
        for item in document_chunks:
            item["document_title"] = document["title"]
            chunks.append(item)
    rank = bm25_ranker([item["index_text"] for item in chunks])
    context_tokens = [count_tokens(item["text"]) for item in chunks]
    normalized_contexts = [normalize(item["text"]) for item in chunks]
    unique_context_indexes = list(
        {
            (item["document_title"], item["start"], item["end"]): index
            for index, item in enumerate(chunks)
        }.values()
    )

    rankings, failures = [], []
    for question in questions:
        normalized_answers = [normalize(answer) for answer in question["answers"]]
        hit = lambda index: any(answer in normalized_contexts[index] for answer in normalized_answers)
        ranked, seen = [], set()
        for index in rank(question["question"]):
            item = chunks[index]
            identity = (item["document_title"], item["start"], item["end"])
            if identity in seen:
                continue
            seen.add(identity)
            ranked.append(index)
            if len(ranked) == 10:
                break
        relevances = [int(hit(index)) for index in ranked]
        total_relevant = sum(hit(index) for index in unique_context_indexes)
        rankings.append(
            {
                "relevances": relevances,
                "total_relevant": total_relevant,
                "context_tokens": [context_tokens[index] for index in ranked],
            }
        )
        if not any(relevances[:5]) and len(failures) < 4:
            top = chunks[ranked[0]]
            failures.append(
                {
                    "question": question["question"],
                    "gold_answer": question["answers"][0],
                    "gold_document": question["document_title"],
                    "retrieved_document": top["document_title"],
                    "retrieved_chunk": top["text"],
                }
            )

    metrics = aggregate_metrics(rankings)
    recalls = [metrics[f"recall_at_{k}"] for k in KS]
    assert recalls == sorted(recalls), f"{strategy['id']} recall@k must be monotone"
    metrics.update(
        {
            "chunks_per_document": statistics.fmean(chunks_per_doc),
            "mean_chunk_tokens": statistics.fmean(context_tokens),
            "median_chunk_tokens": statistics.median(context_tokens),
            "token_counts_estimated": token_counts_estimated,
        }
    )
    return {**strategy, "metrics": metrics}, failures


def inspector_data(documents: list[dict]) -> list[dict]:
    output = []
    for document in documents[:2]:
        # ponytail: cap inspector payload; add full articles if boundary inspection needs them.
        sample = {"title": document["title"] + " (6,000-character excerpt)", "text": document["text"][:6000]}
        sample["strategies"] = {}
        for strategy in STRATEGIES:
            items = build_chunks(strategy["id"], sample)
            sample["strategies"][strategy["id"]] = [
                {"start": item["start"], "end": item["end"], "index_start": item["index_start"], "index_end": item["index_end"]}
                for item in items
            ]
        output.append(sample)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="run metric assertions without an API key")
    args = parser.parse_args()
    metric_selftest()
    if args.selftest:
        print("metric self-check: PASS")
        return

    raw = load_squad()
    documents, questions = sample_corpus(raw)
    if len(documents) != ARTICLE_COUNT or len(questions) != QUESTION_COUNT:
        raise SystemExit(f"sample expected {ARTICLE_COUNT} articles and {QUESTION_COUNT} questions")

    count_tokens, token_counts_estimated, token_count_method = token_counter()
    print(f"token counts: {token_count_method}", flush=True)

    results, failures = [], {}
    for strategy in STRATEGIES:
        print(f"building {strategy['name']}", flush=True)
        result, strategy_failures = score_strategy(strategy, documents, questions, count_tokens, token_counts_estimated)
        results.append(result)
        failures[strategy["id"]] = strategy_failures

    winner = max(
        results,
        key=lambda result: (
            result["metrics"]["ndcg_at_10"],
            -(result["metrics"]["context_tokens_to_90"] or math.inf),
        ),
    )["id"]
    best_recall = max(results, key=lambda result: result["metrics"]["recall_at_10"])
    worst_recall = min(results, key=lambda result: result["metrics"]["recall_at_10"])
    meta = {
        "source": "SQuAD v1.1 dev",
        "source_url": SQUAD_URL,
        "license": "CC BY-SA 4.0",
        "article_count": len(documents),
        "question_count": len(questions),
        "retriever": "bm25",
        "date": date.today().isoformat(),
        "seed": SEED,
        "token_counting": {"method": token_count_method, "estimated": token_counts_estimated},
        "hit_criterion": "case-insensitive normalized gold-answer string containment",
        "parameters": {
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "fixed_size_chars": 500,
            "overlap_10_chars": 50,
            "overlap_25_chars": 125,
            "sentence_window_neighbors": 2,
            "parent_chars": 1000,
            "child_chars": 250,
            "semantic_breakpoint_percentile": 80,
            "large_fixed_size_chars": 1000,
        },
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "results.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "retriever": "bm25",
                "bm25": {"k1": BM25_K1, "b": BM25_B},
                "article_count": len(documents),
                "question_count": len(questions),
                "seed": SEED,
                "summary": {
                    "best_recall_at_10_strategy": best_recall["id"],
                    "best_recall_at_10": best_recall["metrics"]["recall_at_10"],
                    "worst_recall_at_10_strategy": worst_recall["id"],
                    "worst_recall_at_10": worst_recall["metrics"]["recall_at_10"],
                    "recall_at_10_spread_percentage_points": round(
                        (best_recall["metrics"]["recall_at_10"] - worst_recall["metrics"]["recall_at_10"]) * 100,
                        6,
                    ),
                },
                "strategies": results,
            },
            indent=2,
        )
    )
    (DATA / "examples.json").write_text(
        json.dumps({"failures": failures, "documents": inspector_data(documents)}, indent=2)
    )
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote results for {len(documents)} articles, {len(questions)} questions; winner: {winner}")


if __name__ == "__main__":
    main()
