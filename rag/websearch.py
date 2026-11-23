"""Opt-in web search through your own SearXNG instance.

Never called automatically: the UI shows the exact query and the user must approve it.
Results are untrusted text and are passed to the model in separate <web_source> tags.
"""
import html
import json
import logging
import re

import requests

log = logging.getLogger("rag.web")

MAX_QUERY = 200
MAX_SNIPPET = 1200
MAX_BYTES = 2_000_000


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    text = re.sub(r"[​-‏⁠﻿]", "", text)  # zero-width / hidden characters
    return re.sub(r"\s+", " ", text).strip()


def search(base_url: str, query: str, max_results: int = 3, timeout: float = 15) -> list[dict]:
    """Returns hits shaped like local ones: {"id", "name", "source", "heading", "text", "web": True}."""
    query = _clean(query)[:MAX_QUERY]
    if not query:
        return []
    r = requests.get(base_url.rstrip("/") + "/search", params={"q": query, "format": "json"},
                     timeout=timeout, stream=True, headers={"Accept": "application/json"})
    r.raise_for_status()
    body = r.raw.read(MAX_BYTES + 1, decode_content=True)
    if len(body) > MAX_BYTES:
        raise ValueError("SearXNG response too large")
    results = json.loads(body).get("results", [])
    hits = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "")
        if not url.startswith(("http://", "https://")):
            continue
        text = _clean(str(item.get("content") or ""))[:MAX_SNIPPET]
        if not text:
            continue
        hits.append({"id": f"web:{len(hits)}", "name": _clean(str(item.get("title") or ""))[:200] or url,
                     "source": url, "heading": "web", "text": text, "web": True, "score": 0.0})
        if len(hits) >= max_results:
            break
    return hits
