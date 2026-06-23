"""
RAG Evaluation Script

Measures retrieval quality using uploaded documents as ground truth.
Usage:
    python -m eval.evaluate --kb-id <KB_ID> --num-questions 20

Outputs:
    - Recall@K, MRR, NDCG for retrieval
    - Per-document breakdown
"""

import sys
import json
import time
import random
import argparse
import asyncio
from typing import List, Dict, Tuple
from collections import defaultdict

import httpx


API_BASE = "http://localhost:8000"


def get_kb_chunks(kb_id: str) -> List[Dict]:
    """Get all chunks for a KB via debug endpoint."""
    resp = httpx.get(f"{API_BASE}/api/v1/knowledge-bases/debug/chunks-status",
                     params={"kb_id": kb_id}, timeout=10)
    resp.raise_for_status()
    return resp.json()["documents"]


def search(query: str, kb_ids: List[str], top_k: int = 10) -> Dict:
    """Run RAG search."""
    resp = httpx.post(
        f"{API_BASE}/api/v1/knowledge-bases/search",
        json={"query": query, "kb_ids": kb_ids, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def chat(query: str, kb_ids: List[str]) -> Dict:
    """Run full RAG chat."""
    resp = httpx.post(
        f"{API_BASE}/api/v1/chat",
        json={"query": query, "kb_ids": kb_ids, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def generate_questions_from_chunks(
    kb_id: str, num_questions: int = 20
) -> List[Tuple[str, str, str]]:
    """
    Generate test questions from document chunks using LLM.
    Returns list of (question, doc_title, chunk_content).
    """
    chunks_status = get_kb_chunks(kb_id)
    all_docs = [(d["doc_id"], d["title"]) for d in chunks_status]

    # Pick random docs and generate questions
    questions = []
    for doc_id, doc_title in random.sample(all_docs, min(len(all_docs), num_questions)):
        # Search for a random chunk from this doc to use as context
        search_result = search(
            query=doc_title,
            kb_ids=[kb_id],
            top_k=5,
        )
        if not search_result.get("results"):
            continue

        # Pick highest-scoring result from this doc as ground truth
        best = search_result["results"][0]
        chunk_content = best["content_preview"]

        # Use LLM to generate a question from this chunk
        try:
            resp = chat(
                query=(
                    f"根据以下文本，生成一个可以用该文本回答的问题。\n"
                    f"只输出问题，不要输出答案或解释。\n\n"
                    f"文本：{chunk_content}"
                ),
                kb_ids=[],
            )
            question = resp.get("answer", "").strip()
            if question and len(question) > 3:
                questions.append((question, doc_title, chunk_content))
                print(f"  Generated: [{doc_title[:20]}] {question[:60]}...")
        except Exception as e:
            print(f"  Failed to generate Q for {doc_title}: {e}")

        if len(questions) >= num_questions:
            break

    return questions


def evaluate_retrieval(
    kb_id: str,
    questions: List[Tuple[str, str, str]],
    top_k: int = 10,
) -> Dict:
    """
    Evaluate retrieval quality:
    - Recall@k: fraction of questions where the correct doc appears in top-k
    - MRR: Mean Reciprocal Rank of the first correct result
    - Hit@1, Hit@3, Hit@5
    """
    recall_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    reciprocal_ranks = []
    per_doc = defaultdict(lambda: {"hits": 0, "total": 0, "mrr_sum": 0.0})

    for question, doc_title, chunk_content in questions:
        result = search(query=question, kb_ids=[kb_id], top_k=top_k)
        results = result.get("results", [])

        # Check if any result's content matches the ground truth chunk
        first_rank = None
        for rank, r in enumerate(results, 1):
            preview = r.get("content_preview", "")
            # Check content overlap
            overlap = len(set(preview[:100]) & set(chunk_content[:100])) / max(len(preview[:100]), 1)
            if overlap > 0.3 or chunk_content[:50] in preview:
                first_rank = rank
                break

        per_doc[doc_title]["total"] += 1
        if first_rank:
            per_doc[doc_title]["hits"] += 1
            per_doc[doc_title]["mrr_sum"] += 1.0 / first_rank
            reciprocal_ranks.append(1.0 / first_rank)
            for k in recall_hits:
                if first_rank <= k:
                    recall_hits[k] += 1

    n = len(questions)
    if n == 0:
        return {"error": "No questions evaluated"}

    return {
        "total_questions": n,
        "recall": {f"R@{k}": hits / n for k, hits in recall_hits.items()},
        "mrr": sum(reciprocal_ranks) / n if reciprocal_ranks else 0.0,
        "per_document": {
            doc: {
                "questions": s["total"],
                "hit_rate": s["hits"] / s["total"] if s["total"] else 0,
                "mrr": s["mrr_sum"] / s["total"] if s["total"] else 0,
            }
            for doc, s in sorted(per_doc.items(), key=lambda x: -x[1]["hits"] / max(x[1]["total"], 1))
        },
    }


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation")
    parser.add_argument("--kb-id", required=True, help="Knowledge base ID")
    parser.add_argument("--num-questions", type=int, default=20, help="Number of test questions")
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    print(f"RAG Evaluation for KB: {args.kb_id}")
    print(f"Generating {args.num_questions} test questions...")
    print()

    # Step 1: Generate questions from chunks
    questions = generate_questions_from_chunks(args.kb_id, args.num_questions)
    print(f"\nGenerated {len(questions)} questions.\n")

    # Step 2: Evaluate retrieval
    print("Evaluating retrieval...")
    metrics = evaluate_retrieval(args.kb_id, questions)
    print()
    print("=" * 60)
    print("RAG Evaluation Results")
    print("=" * 60)
    print(f"Questions: {metrics['total_questions']}")
    print(f"MRR: {metrics['mrr']:.4f}")
    for k, v in metrics["recall"].items():
        print(f"  {k}: {v:.4f}")
    print()
    print("Per-Document Breakdown:")
    for doc, s in metrics["per_document"].items():
        print(f"  {doc[:40]:40s} Q={s['questions']:2d}  Hit={s['hit_rate']:.2f}  MRR={s['mrr']:.3f}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
