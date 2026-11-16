"""Optional cross-encoder reranking with FlashRank (ONNX, no PyTorch).

Ollama has no rerank endpoint, so this runs in-process. Install with:
    pip install -r requirements-rerank.txt
The model (~34 MB) is downloaded once on first use, then everything is local.
"""
import logging
import threading

log = logging.getLogger("rag.rerank")


class Reranker:
    def __init__(self, model: str, cache_dir: str, mode: str = "auto"):
        self.model, self.cache_dir = model, cache_dir
        self.enabled = False
        self._ranker = None
        self._lock = threading.Lock()
        if mode == "off":
            return
        try:
            import flashrank  # noqa: F401

            self.enabled = True
        except ImportError:
            if mode == "on":
                raise SystemExit("rerank: on, but flashrank is not installed. "
                                 "Run: pip install -r requirements-rerank.txt")
            log.info("Reranker not installed (optional): pip install -r requirements-rerank.txt")

    def rerank(self, query: str, passages: list[dict]) -> list[float] | None:
        """Scores (0-1, higher is better) aligned with `passages`, or None if unavailable."""
        if not self.enabled or not passages:
            return None
        try:
            from flashrank import Ranker, RerankRequest

            with self._lock:
                if self._ranker is None:
                    self._ranker = Ranker(model_name=self.model, cache_dir=self.cache_dir)
                ranked = self._ranker.rerank(RerankRequest(
                    query=query, passages=[{"id": i, "text": p["text"]} for i, p in enumerate(passages)]))
            scores = [0.0] * len(passages)
            for r in ranked:
                scores[int(r["id"])] = float(r["score"])
            return scores
        except Exception as e:  # e.g. the model could not be downloaded
            log.warning("Reranking failed, continuing without it: %s", e)
            self.enabled = False
            return None
