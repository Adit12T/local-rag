"""Measure retrieval quality on your own questions.

    python -m rag.eval eval/golden.yaml

The golden file is a list of questions with the file that should be found:

    - question: What is the Kafka retention for the orders topic?
      expect: kafka.md            # substring of the expected file path

Prints hit@1 / hit@5 / hit@10 and MRR for each retrieval mode, so a settings
change (chunk size, model, reranker) can be judged by numbers instead of feel.
"""
import argparse
import logging
import statistics
from pathlib import Path

import yaml

from .__main__ import make_llm
from .config import load_config
from .index import Indexer
from .ocr import Ocr
from .reranker import Reranker


def evaluate(indexer: Indexer, golden: list[dict], mode: str, rerank: bool) -> dict:
    ranks, top_scores = [], []
    for item in golden:
        result = indexer.search(item["question"], k=10, mode=mode, rerank=rerank)
        paths = [h["source"] for h in result["hits"]]
        ranks.append(next((i for i, p in enumerate(paths, 1) if item["expect"] in p), None))
        if result["hits"]:
            top_scores.append(result["hits"][0]["score"])
    n = len(golden)
    return {
        "hit@1": sum(r == 1 for r in ranks if r) / n,
        "hit@5": sum(r <= 5 for r in ranks if r) / n,
        "hit@10": sum(r <= 10 for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "min_top_score": min(top_scores, default=0),
        "median_top_score": statistics.median(top_scores) if top_scores else 0,
        "missed": [g["question"] for g, r in zip(golden, ranks) if r is None],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval against a golden question set")
    parser.add_argument("golden", nargs="?", default="eval/golden.yaml")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_config(args.config)
    golden = yaml.safe_load(Path(args.golden).read_text())
    indexer = Indexer(cfg, make_llm(cfg), Reranker(cfg.rerank_model, str(cfg.data_dir / "models"), cfg.rerank),
                      Ocr(cfg.ocr))
    print("Syncing index:", indexer.sync())

    runs = [("dense", False), ("keyword", False), ("hybrid", False)]
    if indexer.reranker.enabled:
        runs.append(("hybrid", True))
    print(f"\n{len(golden)} questions\n")
    print(f"{'mode':<18}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'MRR':>8}{'min top':>10}{'median top':>12}")
    last = None
    for mode, rerank in runs:
        last = r = evaluate(indexer, golden, mode, rerank)
        name = mode + (" + rerank" if rerank else "")
        print(f"{name:<18}{r['hit@1']:>8.2f}{r['hit@5']:>8.2f}{r['hit@10']:>8.2f}{r['mrr']:>8.2f}"
              f"{r['min_top_score']:>10.3f}{r['median_top_score']:>12.3f}")
    if last and last["missed"]:
        print("\nNot found by the last mode:")
        for q in last["missed"]:
            print("  -", q)
    print("\n'min top' is the lowest best-score among your answerable questions: set the abstain\n"
          "threshold (rerank_min_score when reranking) safely below it.")


if __name__ == "__main__":
    main()
