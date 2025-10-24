import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List, Tuple
from langchain_core.documents import Document

# Tune as needed
BOTTOM_RATIO = 0.85        # bottom 15% of page likely contains footnotes
SMALL_FONT_RATIO = 0.85    # footnote font typically <= 85% of body median
MAX_FOOTNOTE_CHARS = 500   # truncate very long footnotes (keeps index lean)


def _page_spans(page) -> List[Dict]:
    """Return all text spans with bbox/font size (low-level)."""
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                yield {
                    "text": span.get("text", ""),
                    "size": float(span.get("size", 0.0)),
                    "bbox": span.get("bbox", [0, 0, 0, 0]),
                    "font": span.get("font", ""),
                }


def _group_lines(spans: List[Dict], y_tol: float = 2.0) -> List[str]:
    """
    Rebuild lines from spans to avoid smashed words when joining raw spans.
    Groups spans with similar Y (within y_tol), sorts by Y then X.
    """
    spans = sorted(spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))
    lines: List[str] = []
    cur: List[str] = []
    cur_y = None

    for s in spans:
        txt = s["text"]
        if not txt or not txt.strip():
            continue
        y0 = s["bbox"][1]
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append(txt)
            cur_y = y0 if cur_y is None else (cur_y + y0) / 2
        else:
            lines.append(" ".join(cur))
            cur, cur_y = [txt], y0

    if cur:
        lines.append(" ".join(cur))
    return lines


def _parse_footnotes(foot_blob: str) -> Dict[str, str]:
    """
    Extract {number -> text} from a blob like:
      '107. text ... 108) next note ...'
    Captures text until the next numbered note.
    """
    foot_map: Dict[str, str] = {}
    if not foot_blob.strip():
        return foot_map

    # Primary: anchors like "107." or "107)"
    anchors = list(re.finditer(r"(?P<n>\d{1,4})[.)]\s+", foot_blob))
    if anchors:
        for idx, m in enumerate(anchors):
            n = m.group("n")
            start = m.end()
            end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(foot_blob)
            txt = foot_blob[start:end].strip()
            txt = re.sub(r"\s{2,}", " ", txt)
            if txt:
                foot_map[n] = txt

    # Fallback: sometimes punctuation is missing (e.g., "107 text ... 108 text ...")
    if not foot_map:
        anchors = list(re.finditer(r"(?P<n>\d{1,4})\s+", foot_blob))
        for idx, m in enumerate(anchors):
            n = m.group("n")
            start = m.end()
            next_m = re.search(r"\b\d{1,4}[.)]?\s+", foot_blob[start:])
            end = start + next_m.start() if next_m else len(foot_blob)
            txt = foot_blob[start:end].strip()
            txt = re.sub(r"\s{2,}", " ", txt)
            if txt:
                foot_map[n] = txt

    return foot_map


def _shorten(txt: str, limit: int = MAX_FOOTNOTE_CHARS) -> str:
    """Truncate long footnotes to keep chunks compact (full text still derivable from PDF)."""
    return txt if len(txt) <= limit else txt[:limit].rstrip() + "..."


def _insert_inline_markers(body_text: str, foot_map: Dict[str, str]) -> str:
    """
    Turn e.g. 'concluded107.' into 'concluded[fn107].'
    Only if 107 exists in foot_map. No space allowed before the number
    (to avoid 's 107', 'at 107'). Ignore bracketed numbers like [107].
    """
    if not foot_map:
        return body_text

    # Require no space before number; allow letters, ')' or ']' before it.
    # Examples matched: 'word107', 'word)107', 'word]107'
    pattern = r"(?P<pre>[A-Za-z\)\]])(?P<n>\d{1,4})(?!\d)"

    def repl(m: re.Match) -> str:
        n = m.group("n")
        return f"{m.group('pre')}[fn{n}]" if n in foot_map else m.group(0)

    return re.sub(pattern, repl, body_text)


def _extract_body_and_footnotes(page, bottom_ratio: float = BOTTOM_RATIO,
                                small_font_ratio: float = SMALL_FONT_RATIO) -> Tuple[str, Dict[str, str]]:
    """
    Heuristics:
      - Footnotes tend to be near the bottom (y0 > bottom_ratio * page_height)
      - and use a smaller font than body (<= small_font_ratio * median body size)
    """
    h = page.rect.height
    spans = list(_page_spans(page))
    if not spans:
        return "", {}

    # Estimate body font size using spans above bottom area
    body_sizes = [s["size"] for s in spans if s["bbox"][1] < h * bottom_ratio and (s["text"] or "").strip()]
    body_med = sorted(body_sizes)[len(body_sizes) // 2] if body_sizes else 10.0

    body_spans: List[Dict] = []
    foot_spans: List[Dict] = []
    for s in spans:
        txt = s["text"] or ""
        if not txt.strip():
            continue
        y0 = s["bbox"][1]
        is_bottom = y0 > h * bottom_ratio
        is_small = s["size"] <= body_med * small_font_ratio
        (foot_spans if (is_bottom and is_small) else body_spans).append(s)

    body_text = " ".join(_group_lines(body_spans)).strip()
    foot_blob = " ".join(_group_lines(foot_spans)).strip()

    foot_map = _parse_footnotes(foot_blob)
    return body_text, foot_map


def load_pdfs_footnote_aware(pdf_dir: str) -> List[Document]:
    """
    Scan a folder of PDFs and return one Document per page with:
      - page body (with inline [fnN] markers where resolvable)
      - appended [FOOTNOTES] block (each note truncated to MAX_FOOTNOTE_CHARS)
      - metadata: source filename, file path, 0-based 'page', 1-based 'page_display', raw 'footnotes' map
    """
    docs: List[Document] = []

    for p in Path(pdf_dir).rglob("*.pdf"):
        try:
            doc = fitz.open(p)
        except Exception:
            continue

        try:
            for i, page in enumerate(doc):
                body, fnotes = _extract_body_and_footnotes(page)
                if not body and not fnotes:
                    continue

                body2 = _insert_inline_markers(body, fnotes)

                foot_block = ""
                if fnotes:
                    pairs = [f"[{k}] {_shorten(v)}" for k, v in sorted(fnotes.items(), key=lambda kv: int(kv[0]))]
                    foot_block = "\n\n[FOOTNOTES]\n" + "\n".join(pairs)

                text = (body2 + foot_block).strip()

                meta = {
                    "source": p.name,
                    "file_path": str(p),
                    "page": i,               # 0-based
                    "page_display": i + 1,   # 1-based (nice for humans)
                    "footnotes": fnotes,     # raw mapping for audit
                }
                docs.append(Document(page_content=text, metadata=meta))
        finally:
            doc.close()

    return docs
