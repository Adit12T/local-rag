"""A tiny stand-in for the Ollama HTTP API, for end-to-end and UI tests without models.

    python tests/fake_ollama.py 11500

Implements /api/embed, /api/chat (streaming, `format` structured output, `tools`) and
/api/generate (Yes/No grounding checks), with deterministic word-overlap behaviour.
"""
import hashlib
import json
import math
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STOP = set("a an and are as at be by do does for from how in is it of on or the this to was what when where which "
           "who why with you your our we i".split())


def words(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


def vec(t):
    v = [0.0] * 256
    for w in words(t):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
    n = math.sqrt(sum(x * x for x in v)) or 1
    return [x / n for x in v]


def sources_of(text):
    return re.findall(r'<(?:web_)?source id="(\d+)"[^>]*>\n(.*?)\n</(?:web_)?source>', text, re.S)


def structured(props, user):
    if props == {"query", "route", "sub_queries"}:
        q = user.split("Last message: ", 1)[-1]
        m = re.search(r"compare (.+?) (?:and|with) (.+?)\??$", q, re.I)
        if m:
            return {"query": q, "route": "comparative", "sub_queries": [m.group(1), m.group(2)]}
        return {"query": q, "route": "lookup", "sub_queries": []}
    if props == {"query"}:
        return {"query": user.split("Last message: ", 1)[-1]}
    if props == {"queries"}:
        return {"queries": [" ".join(words(user.removeprefix("Query: "))[:3])]}
    if props == {"sufficient", "missing", "next_query"}:
        return {"sufficient": True, "missing": "", "next_query": ""}
    if props == {"passage"}:
        return {"passage": user}
    if props == {"results"}:
        srcs = dict(sources_of(user))
        out = []
        for k, sent in re.findall(r"^S(\d+): (.*)$", user, re.M):
            ws = set(words(re.sub(r"\[\d+\]", "", sent)))
            sup = [int(i) for i, t in srcs.items() if ws and len(ws & set(words(t))) / len(ws) >= 0.7]
            out.append({"id": int(k), "supported": bool(sup), "sources": sup})
        return {"results": out}
    return {}


def answer_for(messages):
    user = messages[-1]["content"] if "previous answer" not in messages[-1]["content"] else messages[-2]["content"]
    srcs = sources_of(user)
    if not srcs:
        return "I could not find that in the indexed files."
    first = re.split(r"(?<=[.!?])\s+", re.sub(r"^#.*\n", "", srcs[0][1]).strip())[0]
    text = f"{first.rstrip('.')} [1]."
    if len(srcs) > 1 and "compare" in user.lower():
        second = re.split(r"(?<=[.!?])\s+", re.sub(r"^#.*\n", "", srcs[1][1]).strip())[0]
        text += f" {second.rstrip('.')} [2]."
    if "embellish" in user.lower() and "previous answer" not in messages[-1]["content"]:
        text += " It was also approved by the board in a meeting on Mars [1]."
    return text


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/embed":
            return self._json({"embeddings": [vec(t) for t in body["input"]]})
        if self.path == "/api/generate":
            m = re.match(r"Document: (.*)\nClaim: (.*)", body.get("prompt", ""), re.S)
            ok = bool(m) and (lambda ws, doc: bool(ws) and len(ws & doc) / len(ws) >= 0.7)(
                set(words(m.group(2))), set(words(m.group(1))))
            return self._json({"response": "Yes" if ok else "No"})
        if self.path == "/api/chat":
            msgs = body["messages"]
            if body.get("format"):
                props = set(body["format"].get("properties", {}))
                return self._json({"message": {"role": "assistant",
                                               "content": json.dumps(structured(props, msgs[-1]["content"]))}})
            if body.get("tools"):
                if any(m.get("role") == "tool" for m in msgs):
                    return self._json({"message": {"role": "assistant", "content": "DONE"}})
                q = msgs[1]["content"].split("Question: ", 1)[-1].split("\n")[0]
                return self._json({"message": {"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "search", "arguments": {"query": q}}}]}})
            text = answer_for(msgs)
            if not body.get("stream", True):
                return self._json({"message": {"role": "assistant", "content": text}})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in re.findall(r"\S+\s*", text):
                self.wfile.write((json.dumps({"message": {"content": piece}, "done": False}) + "\n").encode())
            self.wfile.write((json.dumps({"message": {"content": ""}, "done": True}) + "\n").encode())
            return
        self._json({"error": "not found"}, 404)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 11500), Handler).serve_forever()
