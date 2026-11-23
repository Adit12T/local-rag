"""Thin client for a local Ollama server (embeddings + chat)."""
import json
import logging
import re
from typing import Iterator

import requests

log = logging.getLogger("rag.ollama")

# Some embedding models are trained with task prefixes and retrieve worse without them.
# (query_prefix, document_prefix), matched on the start of the model name.
KNOWN_PREFIXES = {
    "nomic-embed": ("search_query: ", "search_document: "),
    "mxbai-embed": ("Represent this sentence for searching relevant passages: ", ""),
    "qwen3-embedding": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ", ""),
}


_SCORE = re.compile(r"<score>\s*(yes|no)\s*</score>", re.I)
_BARE = re.compile(r"^\s*(yes|no)\b", re.I)


def guardian_risk(text: str) -> bool:
    """Parse a Granite Guardian verdict. True when the risk is present (for groundedness: unsupported)."""
    m = _SCORE.search(text) or _BARE.search(text)
    if not m:
        raise OllamaError(f"unexpected Granite Guardian output: {text[:80]!r}")
    return m.group(1).lower() == "yes"


class OllamaError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str, embed_model: str, chat_model: str, query_prefix: str | None = None,
                 doc_prefix: str | None = None, num_ctx: int = 8192, temperature: float = 0.1):
        self.url = url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        auto_q, auto_d = next((v for k, v in KNOWN_PREFIXES.items() if embed_model.startswith(k)), ("", ""))
        self.query_prefix = auto_q if query_prefix is None else query_prefix
        self.doc_prefix = auto_d if doc_prefix is None else doc_prefix
        self.options = {"num_ctx": num_ctx, "temperature": temperature}

    @property
    def signature(self) -> str:
        """Anything that makes stored vectors incomparable; a change triggers a re-index."""
        return f"{self.embed_model}|{self.query_prefix}|{self.doc_prefix}"

    def _post(self, route: str, payload: dict, stream: bool = False):
        try:
            r = requests.post(self.url + route, json=payload, stream=stream, timeout=600)
        except requests.ConnectionError as e:
            raise OllamaError(f"Cannot reach Ollama at {self.url}. Is it running? (`ollama serve`)") from e
        if r.status_code == 404:
            model = payload.get("model")
            raise OllamaError(f"Model '{model}' not found. Run: ollama pull {model}")
        if not r.ok:
            raise OllamaError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r

    def _embed(self, texts: list[str], batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            payload = {"model": self.embed_model, "input": texts[i : i + batch], "truncate": False,
                       "keep_alive": "30m"}
            try:
                r = self._post("/api/embed", payload)
            except OllamaError as e:
                if "not found" in str(e) or "Cannot reach" in str(e):
                    raise
                # Most likely a chunk longer than the model's context window.
                log.warning("Embedding without truncation failed (%s); retrying with truncation", e)
                r = self._post("/api/embed", {**payload, "truncate": True})
            out.extend(r.json()["embeddings"])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]

    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {"model": self.chat_model, "messages": messages, "stream": True, "options": self.options}
        r = self._post("/api/chat", payload, stream=True)
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                raise OllamaError(data["error"])
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece

    def chat_json(self, messages: list[dict], schema: dict) -> dict:
        """One non-streaming call constrained to a JSON schema (structured output)."""
        payload = {"model": self.chat_model, "messages": messages, "stream": False, "format": schema,
                   "options": {**self.options, "temperature": 0}}
        out = json.loads(self._post("/api/chat", payload).json()["message"]["content"])
        if not isinstance(out, dict):
            raise OllamaError("structured output was not a JSON object")
        return out

    def chat_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """One non-streaming call that may return tool calls (Ollama `tools`)."""
        payload = {"model": self.chat_model, "messages": messages, "stream": False, "tools": tools,
                   "options": {**self.options, "temperature": 0}}
        return self._post("/api/chat", payload).json().get("message", {})

    def guardian_check(self, model: str, document: str, claim: str) -> bool:
        """Grounding check with IBM Granite Guardian (granite4.1-guardian, Apache 2.0). True = supported.

        The criterion id goes in the system message, the source in a `document` message and the claim
        in the assistant message being judged. The model answers `<score> yes </score>` when the risk
        is present, i.e. when the claim is NOT supported by the document.
        """
        payload = {"model": model, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": "groundedness"},
                                # the model's template does not escape documents; double quotes can break it
                                {"role": "document", "content": document.replace('"', "'")},
                                {"role": "assistant", "content": claim}],
                   "options": {"temperature": 0, "num_ctx": self.options["num_ctx"]}}
        text = self._post("/api/chat", payload).json().get("message", {}).get("content", "")
        return not guardian_risk(text)

    def check(self, model: str, document: str, claim: str) -> bool:
        """Grounding check with a fact-checking model such as bespoke-minicheck (answers Yes/No)."""
        payload = {"model": model, "prompt": f"Document: {document}\nClaim: {claim}", "stream": False,
                   "options": {"temperature": 0, "num_ctx": self.options["num_ctx"]}}
        return self._post("/api/generate", payload).json().get("response", "").strip().lower().startswith("yes")
