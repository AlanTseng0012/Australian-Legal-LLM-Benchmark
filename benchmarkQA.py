# qa_generator_plus.py
import os
import re
import time
import string
import shutil
import unicodedata, re
import fitz  # PyMuPDF
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from openpyxl import Workbook
from openai import OpenAI
from semchunk import chunk_raw
from tiktoken import get_encoding
from dotenv import load_dotenv, find_dotenv

# ---------- Config ----------
MAX_TOKENS_PER_CHUNK = 1024
MIN_WORDS_PER_CHUNK = 20
SLEEP_SECS = 6
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Setup OpenAI client
load_dotenv(find_dotenv())
client = OpenAI()
_tokenizer = get_encoding("cl100k_base")

@lru_cache(maxsize=2048)
def token_counter(text: str) -> int:
    return len(_tokenizer.encode(text))

# ---------- PDF text & spans ----------
def safe_get_text(page):
    # be defensive-some PDFs are finicky
    for mode in ("text", None):
        try:
            return page.get_text(mode) if mode else page.get_text()
        except Exception:
            pass
    return page.get_text()

def build_full_text_and_spans(file_path: str):
    """
    Returns
      full_text : str
      spans     : list of (start_idx, end_idx, page_no[1-based])
    """
    doc = fitz.open(file_path)
    pages, spans, cursor = [], [], 0
    for pno, page in enumerate(doc, start=1):
        t = safe_get_text(page)
        pages.append(t)
        start, end = cursor, cursor + len(t)
        spans.append((start, end, pno))
        cursor = end + 1  # +1 for the newline we join with
    full_text = "\n".join(pages)
    doc.close()
    return full_text, spans

def load_semantic_chunks(file_path: str, max_tokens=MAX_TOKENS_PER_CHUNK, print_chunks=False):
    """
    Build semantic chunks and map each to (pages, start, end) in full_text.
    Returns:
      filtered : list[(chunk_text, pages_here:list[int], start:int, end:int)]
      spans    : as above
      full_text: str
    """
    full_text, spans = build_full_text_and_spans(file_path)

    # semantically chunk the entire text
    chunks = chunk_raw(full_text, token_counter=token_counter, max_tokens=max_tokens)

    # locate each chunk within full_text and map to pages
    chunk_with_meta = []
    search_from = 0
    for c in chunks:
        idx = full_text.find(c, search_from)
        if idx == -1:
            idx = full_text.find(c)
        if idx == -1:
            continue
        start, end = idx, idx + len(c)
        pages_here = [pno for (s, e, pno) in spans if not (end <= s or start >= e)]
        chunk_with_meta.append((c, pages_here, start, end))
        search_from = end

    # light filtering
    filtered = [
        (c, pgs, s, e)
        for (c, pgs, s, e) in chunk_with_meta
        if len(c.split()) > MIN_WORDS_PER_CHUNK and "high court" not in c.lower()
    ]

    if print_chunks:
        for i, (c, pgs, _, __) in enumerate(filtered, 1):
            print(f"\n--- Chunk {i} (pages {pgs}) ---\n{c}\n" + "-" * 80)

    return filtered, spans, full_text

# ---------- Case name / references ----------
def extract_case_name(file_path: str) -> str:
    doc = fitz.open(file_path)
    first_page_text = doc[0].get_text()
    doc.close()

    lines = first_page_text.splitlines()
    combined_text = " ".join(lines)
    m = re.search(
        r'([A-Z][A-Za-z&()\s.,]+? v [A-Z][A-Za-z&()\s.,]+?)\s*\[(\d{4})\]\s*HCA\s*(\d{1,3})',
        combined_text, flags=re.IGNORECASE
    )
    if m:
        case_name = m.group(1).strip()
        year = m.group(2)
        number = m.group(3)
        return f"{case_name} [{year}] HCA {number}"

    # Fallback for Acts
    for line in lines:
        m2 = re.search(r'([A-Za-z\s]+Act\s+(?:19|20)\d{2})', line)
        if m2:
            return m2.group(1).strip()

    return "an Australian legal document"

def extract_references_from_pdf(file_path: str):
    """
    Collect lines like "[23] ..." as a dict {23: "citation text ..."}.
    """
    doc = fitz.open(file_path)
    lines = [line.strip() for page in doc for line in page.get_text().split("\n") if line.strip()]
    doc.close()

    reference_dict = {}
    current_key = None
    for line in lines:
        m = re.match(r"\[(\d{1,3})\]\s*(.+)", line)
        if m:
            ref_num = int(m.group(1))
            if len(str(ref_num)) == 4:  # likely a year like [2014]
                continue
            current_key = ref_num
            reference_dict[current_key] = m.group(2).strip()
        elif current_key and line:
            reference_dict[current_key] += " " + line.strip()

    return dict(sorted(reference_dict.items()))

def resolve_references(snippet: str, reference_dict: dict) -> str:
    """
    Replace inline markers [n] with "citation text [n]" so they are self-contained.
    """
    out = snippet
    for ref, citation in reference_dict.items():
        out = out.replace(f"[{ref}]", f"{citation} [{ref}]")
    return out

# ---------- Evidence finder (QA version) ----------
def extract_support_and_page_for_answer(resolved_chunk: str, answer_text: str,
                                        full_text: str, spans: list, chunk_start: int):
    """
    Find 1~2 sentences in the chunk that best support the generated ANSWER.
    - Score by bag-of-words overlap between answer_text and each sentence.
    - Map the first supporting sentence back to a PDF page via spans.
    Returns: (support_quote: str, page_no: int|None)
    """
    # Split chunk into sentences (simple heuristic)
    sentences = re.split(r'(?<=[\.\?\!])\s+', resolved_chunk.strip())
    if not sentences:
        return "", None

    tgt_words = set(w.lower() for w in re.findall(r"\w+", answer_text))
    scored = []
    cursor = 0
    for s in sentences:
        s_words = set(w.lower() for w in re.findall(r"\w+", s))
        score = len(tgt_words & s_words)

        local_idx = resolved_chunk.find(s, cursor)
        if local_idx == -1:
            local_idx = resolved_chunk.find(s)
        abs_start = chunk_start + max(local_idx, 0)
        scored.append((score, s, abs_start))
        if local_idx != -1:
            cursor = local_idx + len(s)

    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [s for sc, s, _ in scored if sc > 0][:2] or [scored[0][1]]

    first_abs = next((abs_start for sc, s, abs_start in scored if s == picks[0]), None)
    page = None
    if first_abs is not None:
        for (s, e, pno) in spans:
            if s <= first_abs < e:
                page = pno
                break

    quote = " ".join(picks)
    return quote, page

# ---------- Prompting ----------
def generate_qa(snippet: str, case_name: str):
    case_ctx = (
        f"This text is from the case *{case_name}*.\n"
        f"IMPORTANT: Use the case name exactly as written: **{case_name}**.\n"
    )
    prompt = f"""# Snippet
{case_ctx}The snippet from an Australian legal document from which you must synthesise a question and an answer is below.

<snippet>
{snippet}
</snippet>

# Format
Return exactly:
# Question
In the case of Insurance Contracts Act 1984, what is the duty imposed on parties to a contract of insurance regarding good faith?

# Answer
{{A concise answer, extracted from the snippet, that stands alone without the snippet.}}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=700,
    )
    text = resp.choices[0].message.content.strip()

    # Parse
    q = a = ""
    m_q = re.search(r"# Question\s*(.+?)\s*# Answer", text, re.DOTALL | re.IGNORECASE)
    m_a = re.search(r"# Answer\s*(.+)$", text, re.DOTALL | re.IGNORECASE)
    if m_q: q = m_q.group(1).strip()
    if m_a: a = m_a.group(1).strip()
    return q, a, text

# ---------- Utils ----------
def sanitize_filename(name, fallback="Australian_legal_document", max_length=100):
    safe_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = ''.join(c for c in name if c in safe_chars).strip()
    if len(cleaned) > max_length:
        return fallback
    return cleaned or fallback

# ---------- Main ----------
if __name__ == "__main__":
    file_path = input("Enter path to your legal PDF file: ").strip()
    START_INDEX = 11
    N_TO_GENERATE = 1

    case_name = extract_case_name(file_path)
    refs = extract_references_from_pdf(file_path)
    chunks_meta, spans, full_text = load_semantic_chunks(file_path, max_tokens=MAX_TOKENS_PER_CHUNK)

    # Excel setup
    wb = Workbook()
    ws = wb.active
    ws.title = "QA"
    ws.append(["ChunkIndex", "Question", "Answer", "Page", "Evidence"])

    made = 0
        # Clean control chars for Excel

    _BAD_XML = re.compile(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F"   # C0
        r"\x7F-\x9F"                     # C1
        r"\u00AD\u180E"                  # soft hyphen, Mongolian sep
        r"\u200B-\u200F"                 # zero-width + bidi marks
        r"\u2028\u2029"                  # line/para separators
        r"\u202A-\u202E"                 # embedding/override
        r"\u2060\u2066-\u2069"          # word-joiner + bidi isolates
        r"\ufeff]"                      # BOM
    )

    def xlsx_safe(s: str) -> str:
        if not isinstance(s, str): return s
        s = unicodedata.normalize("NFKC", s)                  # canonicalize
        s = s.replace("\r\n", "\n").replace("\r", "\n")       # normalize newlines
        s = _BAD_XML.sub("", s)                               # strip known bads
        s = "".join(ch for ch in s if (ch in "\t\n" or unicodedata.category(ch)[0] != "C"))
        return s.replace("\u00A0", " ")                       # NBSP -> space

    def xl_clip(s: str, limit=32767): 
        return s[:limit] if isinstance(s, str) else s

    
    for i, (snippet, pages_here, chunk_start, chunk_end) in enumerate(chunks_meta[START_INDEX:], start=START_INDEX):
        if made >= N_TO_GENERATE:
            break
        if not snippet.strip():
            continue

        print(f"\n--- Chunk {i+1} (pages {pages_here}) ---")
        resolved = resolve_references(snippet, refs)

        try:
            q, a, raw = generate_qa(resolved, case_name)
            if not (q and a):
                print(f"[Skipped chunk {i+1}]: Could not parse Q/A format")
                continue

            # Find supporting evidence & page for the generated ANSWER
            evidence, page_no = extract_support_and_page_for_answer(
                resolved_chunk=resolved,
                answer_text=a,
                full_text=full_text,
                spans=spans,
                chunk_start=chunk_start
            )
            
            q2  = xl_clip(xlsx_safe(q))
            a2  = xl_clip(xlsx_safe(a))
            ev2 = xl_clip(xlsx_safe(evidence))
            ws.append([i + 1, q2, a2, str(page_no or ""), ev2])
            made += 1

        except Exception as e:
            print(f"[Error at chunk {i+1}]: {e}")

        time.sleep(SLEEP_SECS)

    safe = sanitize_filename(case_name)
    out_path = os.path.join(os.path.expanduser("~"), "Desktop", f"{safe}_qa_results.xlsx")

    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        wb.save(tmp.name)
    shutil.move(tmp.name, out_path)
    print(f"\n==== QA results saved to: {out_path} ====")
