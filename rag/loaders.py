"""Turn a file into (heading, text) sections, and sections into chunks.

Structure is kept where the format has one: Markdown is split at headings,
Python at top-level functions/classes, brace languages (Java, Kotlin, JS/TS,
C#, Go, ...) at types and their methods, PDFs per page, Word files at heading
styles. Everything else falls back to a size-based recursive split.
"""
import ast
import logging
import re
from pathlib import Path

log = logging.getLogger("rag.loaders")

MARKDOWN = {".md", ".markdown"}
BRACE_LANGS = {".java", ".kt", ".js", ".jsx", ".ts", ".tsx", ".cs", ".go", ".c", ".h", ".cpp", ".rs",
               ".scala", ".swift"}
CODE = BRACE_LANGS | {".py", ".sql", ".sh", ".gradle", ".xml", ".json", ".yaml", ".yml", ".properties"}
SEP = " › "
MIN_PAGE_TEXT = 20  # a PDF page with fewer characters than this is treated as scanned

Section = tuple[str, str]  # (heading, text)


def load_sections(path: Path, code_limit: int = 2400, ocr=None) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_sections(path, ocr)
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
    if suffix in BRACE_LANGS:
        return brace_sections(text, suffix, code_limit)
    return [("", text)]


def _pdf_sections(path: Path, ocr=None) -> list[Section]:
    from pypdf import PdfReader

    texts = [(page.extract_text() or "") for page in PdfReader(str(path)).pages]
    scanned = [i for i, t in enumerate(texts) if len(t.strip()) < MIN_PAGE_TEXT]
    if scanned and ocr is not None and ocr.enabled:
        try:
            for i, t in ocr.pdf_pages(path, scanned).items():
                texts[i] = t
        except Exception as e:
            log.warning("OCR failed for %s: %s", path, e)
    return [(f"page {i}", t) for i, t in enumerate(texts, 1)]


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


# ---------- brace languages (Java, Kotlin, JS/TS, C#, Go, C/C++, Rust, ...) ----------

_TYPE_WORDS = ("class", "interface", "enum", "record", "object", "struct", "trait", "impl", "namespace")
_TYPE = re.compile(r"\b(" + "|".join(_TYPE_WORDS) + r")\s+([A-Za-z_]\w*)")
_GO_TYPE = re.compile(r"\btype\s+([A-Za-z_]\w*)\s+(struct|interface)\b")
_CALL = re.compile(r"([A-Za-z_]\w*)\s*\(")
_VAR = re.compile(r"\b(?:const|let|var|val)\s+([A-Za-z_]\w*)")
_NOT_NAMES = {"if", "for", "while", "switch", "catch", "synchronized", "return", "new", "func", "function",
              "async", "fun", "fn", "throw", "else", "do", "try", "super", "this", "when", "match", "using"}


def _line_depths(text: str, char_literals: bool = True) -> tuple[list[int], list[bool]] | None:
    """For every line: brace depth at its end, and whether a real `{` opened on it.

    Braces in strings and comments are ignored. Returns None when the braces do not
    balance (unusual syntax): the caller then falls back to size-based splitting.
    """
    depths, opened, seen, depth, state, i, n = [], [], False, 0, None, 0, len(text)
    while i < n:
        c = text[i]
        if c == "\n":
            if state in ("line", '"', "'"):
                state = None
            depths.append(depth)
            opened.append(seen)
            seen = False
        elif state == "line":
            pass
        elif state == "block":
            if c == "*" and text[i + 1 : i + 2] == "/":
                state, i = None, i + 1
        elif state == '"""':
            if text.startswith('"""', i):
                state, i = None, i + 2
        elif state is not None:  # inside ", ' or `
            if c == "\\":
                i += 1
            elif c == state:
                state = None
        elif c == "/" and text[i + 1 : i + 2] == "/":
            state, i = "line", i + 1
        elif c == "/" and text[i + 1 : i + 2] == "*":
            state, i = "block", i + 1
        elif text.startswith('"""', i):
            state, i = '"""', i + 2
        elif c in '"`' or (c == "'" and char_literals):
            state = c
        elif c == "{":
            depth, seen = depth + 1, True
        elif c == "}":
            depth -= 1
            if depth < 0:
                return None
        i += 1
    depths.append(depth)
    opened.append(seen)
    return (depths, opened) if depth == 0 and state in (None, "line") else None


def _declaration_name(body: str) -> str:
    """'class Foo', 'process()' or a variable name for a block; '' if it is not a declaration."""
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"@\w+(?:\([^)]*\))?", "", code)  # annotations / decorators
    header = code.split("{", 1)[0]
    m = _GO_TYPE.search(header)
    if m:
        return f"{m.group(2)} {m.group(1)}"
    m = _TYPE.search(header)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    for m in _CALL.finditer(header):
        if m.group(1) not in _NOT_NAMES:
            return m.group(1) + "()"
    m = _VAR.search(header)
    return m.group(1) if m else ""


def brace_sections(text: str, suffix: str = ".java", limit: int = 2400) -> list[Section]:
    """One section per top-level declaration; types larger than `limit` are split into their members."""
    if len(text) > 2_000_000:
        return [("", text)]
    scanned = _line_depths(text, char_literals=suffix != ".rs")  # Rust lifetimes ('a) are not char literals
    if scanned is None:
        return [("", text)]
    depths, opened = scanned
    lines = text.split("\n")
    out: list[Section] = []

    def walk(start: int, end: int, base: int, parent: str, loose: list[str]):
        """Emit sections for lines[start:end]; `loose` collects fields, imports and other non-blocks."""
        def flush():
            if "\n".join(loose).strip():
                out.append((parent, "\n".join(loose)))
            loose.clear()

        seg_start, has_block = start, False
        for i in range(start, end):
            has_block = has_block or opened[i]
            closes = depths[i] == base and (has_block or lines[i].rstrip().endswith(";"))
            if not closes and i < end - 1:
                continue
            body = "\n".join(lines[seg_start : i + 1])
            name = _declaration_name(body) if has_block else ""
            if not name:
                loose.append(body)
            else:
                flush()
                heading = parent + SEP + name if parent else name
                opening = next((j for j in range(seg_start, i + 1) if depths[j] > base), None)
                if name.startswith(_TYPE_WORDS) and len(body) > limit and opening is not None and opening < i:
                    # big type: its declaration line + fields form one section, each member its own
                    walk(opening + 1, i, base + 1, heading, lines[seg_start : opening + 1])
                else:
                    out.append((heading, body))
            seg_start, has_block = i + 1, False
        flush()

    walk(0, len(lines), 0, "", [])
    return _pack(out, limit // 3)


def _pack(sections: list[Section], small: int) -> list[Section]:
    """Merge runs of small sibling members (getters, one-liners) so they are not one chunk each."""
    def parent(h):
        return h.rsplit(SEP, 1)[0] if SEP in h else ""

    packed: list[list] = []
    for heading, text in sections:
        last = packed[-1] if packed else None
        mergeable = (last and heading and last[0] and parent(last[0]) == parent(heading)
                     and not heading.rsplit(SEP, 1)[-1].startswith(_TYPE_WORDS)
                     and not last[0].rsplit(SEP, 1)[-1].startswith(_TYPE_WORDS)
                     and len(last[1]) + len(text) <= small)
        if mergeable:
            leaf = heading.rsplit(SEP, 1)[-1]
            if last[0].count(", ") < 2:
                last[0] += ", " + leaf
            elif not last[0].endswith("…"):
                last[0] += ", …"
            last[1] += "\n" + text
        else:
            packed.append([heading, text])
    return [(h, t) for h, t in packed]


# ---------- chunking ----------

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


def build_chunks(path: Path, size: int, overlap: int, code_size: int, ocr=None) -> list[Section]:
    """Load a file and return its chunks as (heading, text)."""
    is_code = path.suffix.lower() in CODE
    limit, lap = (code_size, 0) if is_code else (size, overlap)
    out = []
    for heading, text in load_sections(path, code_size, ocr):
        out.extend((heading, piece) for piece in chunk_text(text, limit, lap))
    return out
