"""Evaluate semantic duplicate retrieval against 20 labeled forum queries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.retriever import QuestionRetriever

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "forum_qa.json"
RESULTS_PATH = ROOT / "evaluation_results.csv"
TOP_K = 3

TEST_CASES: list[dict[str, Any]] = [
    {"query": "How do I make an isolated Python environment for one project?", "ground_truth": ["PY001", "PY002", "PY003"]},
    {"query": "How can I capture my active pip dependencies in a file?", "ground_truth": ["PY004", "PY005", "PY006"]},
    {"query": "Can I merge two Python mappings and prefer the right-hand values?", "ground_truth": ["PY007", "PY008", "PY009"]},
    {"query": "How do I undo an unpushed commit without throwing away its edits?", "ground_truth": ["GIT001", "GIT002", "GIT003"]},
    {"query": "What command changes the name of my current Git branch?", "ground_truth": ["GIT004", "GIT005", "GIT006"]},
    {"query": "How can I put unfinished tracked changes aside and restore them later?", "ground_truth": ["GIT007", "GIT008", "GIT009"]},
    {"query": "How can a host machine reach a web port inside a Docker container?", "ground_truth": ["DOCKER001", "DOCKER002", "DOCKER003"]},
    {"query": "Where should a container store database files so they survive removal?", "ground_truth": ["DOCKER004", "DOCKER005", "DOCKER006"]},
    {"query": "Which Docker cleanup command frees disk space from unused resources?", "ground_truth": ["DOCKER007", "DOCKER008", "DOCKER009"]},
    {"query": "Find email addresses that show up repeatedly in my SQL users table.", "ground_truth": ["SQL001", "SQL002", "SQL003"]},
    {"query": "When are database indexes useful and what do they cost?", "ground_truth": ["SQL004", "SQL005", "SQL006"]},
    {"query": "How do I fetch later pages efficiently from a large SQL result set?", "ground_truth": ["SQL007", "SQL008", "SQL009"]},
    {"query": "How do I give a cloud workload only the access it needs?", "ground_truth": ["CLOUD001", "CLOUD002", "CLOUD003"]},
    {"query": "How do I prevent anonymous users from reading my cloud storage bucket?", "ground_truth": ["CLOUD004", "CLOUD005", "CLOUD006"]},
    {"query": "Where should I investigate when cloud application logs are missing?", "ground_truth": ["CLOUD007", "CLOUD008", "CLOUD009"]},
    {"query": "Should a REST API secret be sent in a URL or an HTTPS header?", "ground_truth": ["API001", "API002", "API003"]},
    {"query": "How does a client follow a continuation token to get the next API results?", "ground_truth": ["API004", "API005", "API006"]},
    {"query": "What backoff should I use after the server returns HTTP 429?", "ground_truth": ["API007", "API008", "API009"]},
    {"query": "How can I store and reload a list of installed Python packages?", "ground_truth": ["PY004", "PY005", "PY006"]},
    {"query": "How can I temporarily shelve edits while I switch to another Git branch?", "ground_truth": ["GIT007", "GIT008", "GIT009"]},
]


def evaluate() -> list[dict[str, Any]]:
    with DATA_PATH.open(encoding="utf-8") as handle:
        documents = json.load(handle)
    index = QuestionRetriever(documents)
    rows = []
    for case_number, test_case in enumerate(TEST_CASES, start=1):
        matches = index.search(test_case["query"], top_k=TOP_K)
        retrieved_ids = [item["id"] for item in matches]
        relevant = set(test_case["ground_truth"])
        precision = len(set(retrieved_ids) & relevant) / TOP_K
        rows.append({
            "case": case_number,
            "query": test_case["query"],
            "retrieved_ids": ";".join(retrieved_ids),
            "ground_truth_ids": ";".join(test_case["ground_truth"]),
            "relevant_hits": len(set(retrieved_ids) & relevant),
            "precision_at_3": precision,
        })

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Retrieval backend: {index.backend}")
    for row in rows:
        print(
            f'{row["case"]:02d}. P@3={row["precision_at_3"]:.3f} '
            f'| retrieved={row["retrieved_ids"]} | {row["query"]}'
        )
    mean_precision = sum(row["precision_at_3"] for row in rows) / len(rows)
    print(f"\nMean Precision@3: {mean_precision:.3f} ({mean_precision:.1%})")
    print(f"Results exported to: {RESULTS_PATH}")
    return rows


if __name__ == "__main__":
    evaluate()
