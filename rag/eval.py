"""Measure retrieval and answer quality on your own questions.

    python -m rag.eval eval/golden.yaml             # retrieval: hit@k and MRR per retrieval mode
    python -m rag.eval eval/golden.yaml --answers   # also full answers: plain vs corrective/agentic pipeline

Golden file entries:

    - question: What is the Kafka retention for the orders topic?
      expect: kafka.md               # substring of the file path that holds the answer
      answer_contains: [seven days]  # optional: strings a correct answer must contain
    - question: Compare the retention of orders and payments
      expect: [kafka.md, payments.md] # several files: a multi-part question
    - question: What is our Mars office address?
      unanswerable: true             # nothing in your files answers it: the right reply is "not found"

Answer mode reports, per pipeline: answer accuracy, abstention precision/recall, all-files-found
rate for multi-part questions, sentence support (when verification is on), model calls and
latency. Settings changes can then be judged by numbers instead of feel.
"""
import argparse
import dataclasses
import logging
import statistics
from pathlib import Path

import yaml

from .__main__ import make_llm
from .config import load_config
from .index import Indexer
from .ocr import Ocr
from .pipeline import Pipeline
from .reranker import Reranker


def _expects(item) -> list[str]:
    e = item.get("expect") or []
    return [e] if isinstance(e, str) else list(e)


def evaluate(indexer: Indexer, golden: list[dict], mode: str, rerank: bool) -> dict:
    items = [g for g in golden if _expects(g) and not g.get("unanswerable")]
    ranks, top_scores = [], []
    for item in items:
        result = indexer.search(item["question"], k=10, mode=mode, rerank=rerank)
        paths = [h["source"] for h in result["hits"]]
        found = [next((i for i, p in enumerate(paths, 1) if e in p), None) for e in _expects(item)]
        ranks.append(None if None in found else max(found))  # rank at which every expected file is present
        if result["hits"]:
            top_scores.append(result["hits"][0]["score"])
    n = max(1, len(items))
    return {
        "hit@1": sum(r == 1 for r in ranks if r) / n,
        "hit@5": sum(r <= 5 for r in ranks if r) / n,
        "hit@10": sum(r <= 10 for r in ranks if r) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "min_top_score": min(top_scores, default=0),
        "median_top_score": statistics.median(top_scores) if top_scores else 0,
        "missed": [g["question"] for g, r in zip(items, ranks) if r is None],
    }


def run_answers(pipeline: Pipeline, golden: list[dict]) -> dict:
    rows = []
    for item in golden:
        result = None
        for event in pipeline.run(item["question"], []):
            if event["type"] == "_result":
                result = event
        paths = [s["path"] for s in result["sources"]] if not result["abstain"] else []
        must = [s.lower() for s in item.get("answer_contains", [])]
        v = result["verification"] or {}
        rows.append({
            "unanswerable": bool(item.get("unanswerable")),
            "abstained": result["abstain"],
            "correct": (not result["abstain"] and all(m in result["answer"].lower() for m in must)) if must else None,
            "multi": len(_expects(item)) > 1,
            "all_found": all(any(e in p for p in paths) for e in _expects(item)),
            "support": (1 - v["unsupported"] / v["checked"]) if v.get("checked") else None,
            "calls": result["llm_calls"],
            "ms": result["total_ms"],
            "grade": result["grade"],
        })
    unans = [r for r in rows if r["unanswerable"]]
    abst = [r for r in rows if r["abstained"]]
    scored = [r for r in rows if r["correct"] is not None]
    multi = [r for r in rows if r["multi"]]
    support = [r["support"] for r in rows if r["support"] is not None]
    ms = sorted(r["ms"] for r in rows)

    def share(xs, pred):
        return sum(1 for x in xs if pred(x)) / len(xs) if xs else None

    return {
        "accuracy": share(scored, lambda r: r["correct"]),
        "abstain_recall": share(unans, lambda r: r["abstained"]),
        "abstain_precision": share(abst, lambda r: r["unanswerable"]),
        "multi_all_found": share(multi, lambda r: r["all_found"]),
        "support": statistics.mean(support) if support else None,
        "calls": statistics.mean(r["calls"] for r in rows) if rows else 0,
        "p50_ms": ms[len(ms) // 2] if ms else 0,
        "p95_ms": ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else 0,
        "grades": {g: sum(r["grade"] == g for r in rows) for g in ("correct", "ambiguous", "incorrect")},
    }


def plain(cfg):
    """The same settings with every corrective / agentic / verification feature off."""
    return dataclasses.replace(cfg, corrective=False, router="off", iterate="off", verify="off", agent_mode="off", self_correct=False,
                               expand_neighbors=0, refine_strips=False, hyde=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers) against golden questions")
    parser.add_argument("golden", nargs="?", default="eval/golden.yaml")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--answers", action="store_true", help="also generate answers (slow: calls the chat model)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_config(args.config)
    golden = yaml.safe_load(Path(args.golden).read_text())
    llm = make_llm(cfg)
    reranker = Reranker(cfg.rerank_model, str(cfg.data_dir / "models"), cfg.rerank)
    indexer = Indexer(cfg, llm, reranker, Ocr(cfg.ocr))
    print("Syncing index:", indexer.sync())

    runs = [("dense", False), ("keyword", False), ("hybrid", False)]
    if indexer.reranker.enabled:
        runs.append(("hybrid", True))
    answerable = [g for g in golden if _expects(g) and not g.get("unanswerable")]
    print(f"\nRetrieval: {len(answerable)} answerable questions\n")
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
    print("\n'min top' is the lowest best-score among your answerable questions: keep the 'not found'\n"
          "threshold (rerank_min_score, or min_dense_score without a reranker) safely below it, and set\n"
          "crag_correct_score near the median.")

    if not args.answers:
        return
    print(f"\nAnswers: {len(golden)} questions (this calls the chat model)\n")
    fmt = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    print(f"{'pipeline':<12}{'accuracy':>9}{'abst.rec':>9}{'abst.prec':>10}{'multi':>7}{'support':>9}"
          f"{'calls':>7}{'p50 s':>7}{'p95 s':>7}  grades c/a/i")
    for name, pcfg in (("plain", plain(cfg)), ("full", cfg)):
        indexer.cfg = pcfg
        r = run_answers(Pipeline(pcfg, llm, indexer, reranker), golden)
        g = r["grades"]
        print(f"{name:<12}{fmt(r['accuracy']):>9}{fmt(r['abstain_recall']):>9}{fmt(r['abstain_precision']):>10}"
              f"{fmt(r['multi_all_found']):>7}{fmt(r['support']):>9}{r['calls']:>7.1f}{r['p50_ms'] / 1000:>7.1f}"
              f"{r['p95_ms'] / 1000:>7.1f}  {g['correct']}/{g['ambiguous']}/{g['incorrect']}")
    indexer.cfg = cfg
    print("\naccuracy: answers containing every answer_contains string · abst.rec/prec: 'not found' on\n"
          "unanswerable questions · multi: multi-file questions with every expected file in the sources ·\n"
          "support: share of answer sentences the verifier found in the sources (verification on only)")


if __name__ == "__main__":
    main()
