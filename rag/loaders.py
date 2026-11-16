"""Turn a file into (heading, text) sections, and sections into chunks.

Structure is kept where the format has one: Markdown is split at headings,
Python at top-level functions/classes, PDFs per page, Word files at heading
styles. Everything else falls back to a size-based recursive split.
"""
import ast
import re
from pathlib import Path

MARKDOWN = {".md", ".markdown"}
CODE = {".py", ".java", ".kt", ".js", ".ts", ".tsx", ".jsx", ".go", ".c", ".h", ".cpp", ".cs", ".rs",
        ".sql", ".sh", ".gradle", ".xml", ".json", ".yaml", ".yml", ".properties"}
SEP = " › "

Section = tuple[str, str]  # (heading, text)


def load_sections(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        pages = PdfReader(str(path)).pages
        return [(f"page {i}", page.extract_text() or "") for i, page in enumerate(pages, 1)]
    if suffix == ".docx":
        return _docx_sections(path)
    data = path.read_bytes()
    if b"\x00" in data[:4096]:  # binary file with a text-like extension
        return []
    text = data.decode("utf-8", errors="ignore")
    if suffix in MARKDOWN:
        return markdown_sections(text)
    if suffix == ".py":
        return python_sections(text)
    return [("", text)]


def _docx_sections(path: Path) -> list[Section]:
    import docx

    sections, heading, buf = [], "", []
    for p in docx.Document(str(path)).paragraphs:
        if p.style is not None and (p.style.name or "").startswith("Heading") and p.text.strip():
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = p.text.strip(), [p.text]
        else:
            buf.append(p.text)
    if buf:
        sections.append((heading, "\n".join(buf)))
    return sections


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def markdown_sections(text: str) -> list[Section]:
    """Split at ATX headings (ignoring ones inside code fences); heading = full path 'A › B'."""
    sections, stack, buf, fence = [], [], [], None
    path = ""

    def flush():
        if "".join(buf).strip():
            sections.append((path, "\n".join(buf)))
        buf.clear()

    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            mark = stripped[:3]
            fence = None if fence == mark else (fence or mark)
        m = None if fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = [(lv, t) for lv, t in stack if lv < level] + [(level, m.group(2))]
            path = SEP.join(t for _, t in stack)
        buf.append(line)
    flush()
    return sections


def python_sections(text: str) -> list[Section]:
    """One section per top-level function/class; code in between is grouped as module-level."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [("", text)]
    lines = text.splitlines()
    sections, cursor = [], 0

    def emit(heading, start, end):
        body = "\n".join(lines[start:end])
        if body.strip():
            sections.append((heading, body))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            emit("", cursor, start)
            kind = "class" if isinstance(node, ast.ClassDef) else "def"
            emit(f"{kind} {node.name}", start, node.end_lineno)
            cursor = node.end_lineno
    emit("", cursor, len(lines))
    return sections


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split into ~size-character chunks, preferring paragraph/line/sentence breaks."""
    text = text.strip()
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(path: Path, size: int, overlap: int, code_size: int) -> list[Section]:
    """Load a file and return its chunks as (heading, text)."""
    is_code = path.suffix.lower() in CODE
    limit, lap = (code_size, 0) if is_code else (size, overlap)
    out = []
    for heading, text in load_sections(path):
        out.extend((heading, piece) for piece in chunk_text(text, limit, lap))
    return out
