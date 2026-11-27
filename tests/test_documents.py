"""Document chatbot: upload documents in the UI, pick some, chat only with those."""
import json

from test_pipeline import env, make_client, run, steps  # noqa: F401  (env is a fixture)

from rag.pipeline import NOT_FOUND_SELECTED, Pipeline, is_overview

LONG = "\n\n".join(
    f"## Part {i}\n\nSection {i} of the handbook covers topic number {i} in detail. "
    + ("The refund window is thirty days from delivery. " if i == 0 else "")
    + ("Warranty claims need the original receipt and the serial number. " if i == 9 else "")
    + "Filler sentence about the procedure and the people involved. " * 3
    for i in range(10))


def chat(client, question, files=None, **kw):
    body = {"question": question, "filters": {"files": files or []}, **kw}
    res = client.post("/api/chat", json=body)
    if res.status_code != 200:
        return res, []
    return res, [json.loads(line) for line in res.text.splitlines() if line.strip()]


def upload(client, *files, headers=None):
    return client.post("/api/documents", files=[("files", f) for f in files], headers=headers or {})


def test_upload_indexes_and_lists_the_document(env):
    client = make_client(env["cfg"], env["llm"])
    res = upload(client, ("handbook.md", LONG.encode(), "text/markdown"))
    assert res.status_code == 200
    r = res.json()["results"][0]
    assert r["status"] == "added" and r["chunks"] > 3 and r["name"] == "handbook.md"
    docs = {d["name"]: d for d in client.get("/api/documents").json()["documents"]}
    assert docs["handbook.md"]["uploaded"] is True and docs["kafka.md"]["uploaded"] is False
    assert (env["cfg"].data_dir / "uploads" / "handbook.md").read_text() == LONG


def test_chat_with_selected_documents_only(env):
    client = make_client(env["cfg"], env["llm"])
    path = upload(client, ("handbook.md", LONG.encode(), "text/markdown")).json()["results"][0]["path"]
    _, events = chat(client, "How long is the refund window?", files=[path])
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert sources and {s["name"] for s in sources} == {"handbook.md"}
    # a question about another file finds nothing inside the selection
    _, events = chat(client, "Which Kafka partition stores invoices?", files=[path])
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == NOT_FOUND_SELECTED


def test_summarize_reads_the_whole_document_not_only_matches(env):
    client = make_client(env["cfg"], env["llm"])
    path = upload(client, ("handbook.md", LONG.encode(), "text/markdown")).json()["results"][0]["path"]
    _, events = chat(client, "Summarize this document", files=[path])
    over = [e for e in events if e["type"] == "step" and e["stage"] == "overview"]
    assert over and over[0]["documents"] == 1
    assert not [e for e in events if e["type"] == "step" and e["stage"] == "grade"]  # no keyword search
    previews = " ".join(s["preview"] for s in next(e for e in events if e["type"] == "sources")["sources"])
    assert "Section 0" in previews and "Section 9" in previews  # start and end of the document


def test_overview_question_detection():
    for q in ("Summarize this document", "summarise these", "Give me an overview", "TL;DR?",
              "What is this document about?", "what are the key points", "What's this?"):
        assert is_overview(q), q
    for q in ("How long is the refund window?", "Who signs the warranty claims?", "What is the deploy day?"):
        assert not is_overview(q), q


def test_upload_rejects_bad_files_and_cleans_names(env):
    env["cfg"].max_file_mb = 0.001  # about 1 KB
    client = make_client(env["cfg"], env["llm"])
    res = upload(client, ("../../../evil.md", b"# fine\nshort note about pears", "text/markdown"),
                 ("tool.exe", b"MZ....", "application/octet-stream"),
                 ("empty.md", b"", "text/markdown"),
                 (".env.md", b"x" * 10, "text/markdown"),
                 ("big.md", b"word " * 2000, "text/markdown"))
    by = {r["name"]: r for r in res.json()["results"]}
    up = env["cfg"].data_dir / "uploads"
    assert by["evil.md"]["status"] == "added" and (up / "evil.md").exists()
    assert not (env["tmp"] / "evil.md").exists()
    assert by["tool.exe"]["status"] == "rejected"
    assert by["empty.md"]["status"] == "rejected"
    assert by["big.md"]["status"] == "rejected"
    assert by["env.md"]["status"] == "added"  # leading dot stripped: never a hidden file
    assert not [p for p in up.iterdir() if p.name.startswith(".")]  # no temporary files left behind


def test_same_name_is_not_overwritten(env):
    client = make_client(env["cfg"], env["llm"])
    upload(client, ("notes.md", b"first version about apples", "text/markdown"))
    r = upload(client, ("notes.md", b"second version about plums", "text/markdown")).json()["results"][0]
    assert r["name"] == "notes (1).md"
    assert (env["cfg"].data_dir / "uploads" / "notes.md").read_text() == "first version about apples"


def test_delete_only_removes_uploaded_documents(env):
    client = make_client(env["cfg"], env["llm"])
    path = upload(client, ("notes.md", b"a note about apples", "text/markdown")).json()["results"][0]["path"]
    assert client.delete("/api/documents", params={"path": str(env["docs"] / "kafka.md")}).status_code == 404
    traversal = str(env["cfg"].data_dir / "uploads" / ".." / "index.sqlite")
    assert client.delete("/api/documents", params={"path": traversal}).status_code == 404
    assert (env["docs"] / "kafka.md").exists() and (env["cfg"].data_dir / "index.sqlite").exists()
    assert client.delete("/api/documents", params={"path": path}).status_code == 200
    names = {d["name"] for d in client.get("/api/documents").json()["documents"]}
    assert "notes.md" not in names and "kafka.md" in names


def test_cross_origin_writes_are_refused(env):
    client = make_client(env["cfg"], env["llm"])
    bad = upload(client, ("x.md", b"planted instructions", "text/markdown"), headers={"Origin": "http://evil.example"})
    assert bad.status_code == 403
    assert client.post("/api/chat", json={"question": "hi"}, headers={"Origin": "null"}).status_code == 403
    good = upload(client, ("x.md", b"a real note about pears", "text/markdown"), headers={"Origin": "http://localhost"})
    assert good.status_code == 200


def test_unknown_selected_documents_are_refused(env):
    client = make_client(env["cfg"], env["llm"])
    res, _ = chat(client, "anything", files=["/etc/passwd"])
    assert res.status_code == 400


def test_agent_cannot_read_documents_outside_the_selection(env):
    llm, docs = env["llm"], env["docs"]
    llm.tool_script = [[("read_file", {"path": f"{docs.name}/kafka.md"})]]
    events, result = run(Pipeline(env["cfg"], llm, env["ix"]), "When do deploys happen?",
                         filters={"files": [str(docs / "deploy.md")]})
    assert steps(events, "tool")[0]["ok"] is False
    assert {s["name"] for s in result["sources"]} == {"deploy.md"}


def test_uploads_can_be_turned_off(env):
    env["cfg"].uploads = False
    client = make_client(env["cfg"], env["llm"])
    assert upload(client, ("x.md", b"note", "text/markdown")).status_code == 403
    assert client.get("/api/documents").json()["uploads"] is False
    assert all("uploads" not in str(p) for p in env["cfg"].paths)


def test_upload_only_setup_needs_no_folders(tmp_path):
    from rag.config import load_config
    f = tmp_path / "c.yaml"
    f.write_text(f"paths: []\ndata_dir: {tmp_path / 'd'}\n")
    assert load_config(str(f)).paths == []
