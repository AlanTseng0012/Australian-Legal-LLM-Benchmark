import fitz
import re
import os
import time
import string
import unicodedata
from openai import OpenAI
from openpyxl import Workbook
from semchunk import chunk_raw
from tiktoken import get_encoding
from functools import lru_cache
from dotenv import load_dotenv, find_dotenv

# ---------------- Config ----------------
MAX_TOKENS_PER_CHUNK = 512
MIN_WORDS_PER_CHUNK = 20
MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
SLEEP_SECS = 6

# control how many Qs to make and where to start (1-based)
START_INDEX = 11
N_TO_GENERATE = 2

# --------------- Setup ------------------
load_dotenv(find_dotenv())
client = OpenAI()
_tokenizer = get_encoding("cl100k_base")

@lru_cache(maxsize=2048)
def token_counter(text: str) -> int:
    return len(_tokenizer.encode(text))

# ------------- Chunking -----------------
def load_semantic_chunks(file_path, max_tokens=MAX_TOKENS_PER_CHUNK):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    chunks = chunk_raw(text, token_counter=token_counter, max_tokens=max_tokens)
    return [c for c in chunks if len(c.split()) > MIN_WORDS_PER_CHUNK and "high court" not in c.lower()]

# ------------- Helpers ------------------
def extract_case_name(file_path):
    doc = fitz.open(file_path)
    first_page_text = doc[0].get_text()
    doc.close()
    lines = first_page_text.splitlines()
    combined_text = " ".join(lines)

    m = re.search(r'([A-Z][A-Za-z&()\s.,]+? v [A-Z][A-Za-z&()\s.,]+?)\s*\[(\d{4})\]\s*HCA\s*(\d{1,3})',
                  combined_text, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).strip()} [{m.group(2)}] HCA {m.group(3)}"

    for line in lines:
        m2 = re.search(r'([A-Za-z\s]+Act\s+(?:19|20)\d{2})', line)
        if m2:
            return m2.group(1).strip()
    return "an Australian legal document"

def extract_references_from_pdf(file_path):
    doc = fitz.open(file_path)
    lines = [line.strip() for page in doc for line in page.get_text().split('\n') if line.strip()]
    doc.close()
    reference_dict, current_key = {}, None
    for line in lines:
        m = re.match(r"\[(\d{1,3})\]\s*(.+)", line)
        if m:
            ref_num = int(m.group(1))
            if len(str(ref_num)) == 4:  # skip [2014] etc.
                continue
            current_key = ref_num
            reference_dict[current_key] = m.group(2).strip()
        elif current_key and line:
            reference_dict[current_key] += " " + line.strip()
    return dict(sorted(reference_dict.items()))

def resolve_references(snippet, reference_dict):
    out = snippet
    for ref, citation in reference_dict.items():
        out = out.replace(f"[{ref}]", f"{citation} [{ref}]")
    return out

def sanitize_filename(name, fallback="Australian_legal_document", max_length=100):
    safe_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = ''.join(c for c in name if c in safe_chars).strip()
    if len(cleaned) > max_length:
        return fallback
    return cleaned or fallback

# Excel-safe cleaner (prevents IllegalCharacterError)
_BAD_XML = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F"  # C0
    r"\x7F-\x9F"                    # C1
    r"\u00AD\u180E"                 # soft hyphen, Mongolian sep
    r"\u200B-\u200F"                # zero-width + bidi marks
    r"\u2028\u2029"                 # line/para separators
    r"\u202A-\u202E"                # bidi embeddings/overrides
    r"\u2060\u2066-\u2069"         # word-joiner + bidi isolates
    r"\ufeff]"                     # BOM
)

def xlsx_safe(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _BAD_XML.sub("", s)
    return s.replace("\u00A0", " ")

def xl_clip(s: str, limit=32767) -> str:
    return s[:limit] if isinstance(s, str) else s

# ------------- Generation ---------------
def generate_true_false(snippet, case_name):
    case_context = f"This text is from the case *{case_name}*.\n\n"
    prompt = f"""{case_context}Based on the snippet below, create ONE True/False style question. Then, provide a verbatim excerpt from the snippet that directly supports your answer.

<snippet>
{snippet}
</snippet>

# Format
You must format your response as follows:
<format>
# Statement
{{A factual or interpretive statement based on the snippet.}}

# Answer
{{True or False}}

# Explanation
{{A sentence or passage exactly copied from the snippet that justifies the answer.}}
</format>

# Instructions
- The statement MUST be **false** given the snippet.
- The Answer MUST be exactly **False**.
- Your statement must start with the full case name exactly as provided above (including the square-bracket citation if present).
- The answer must be decontextualised and standalone.
- Use only the snippet as your source.
"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0
    )
    return completion.choices[0].message.content.strip()

# ----------------- MAIN -----------------
if __name__ == "__main__":
    file_path = input("Enter path to your legal PDF file: ").strip()
    case_name = extract_case_name(file_path)
    references = extract_references_from_pdf(file_path)
    chunks = load_semantic_chunks(file_path)

    # Excel setup
    wb = Workbook()
    ws = wb.active
    ws.title = "TrueFalse"
    ws.append(["ChunkIndex", "Statement", "Answer", "Explanation"])

    generated = 0
    # Iterate from START_INDEX (1-based) and stop after N_TO_GENERATE rows
    for i, snippet in enumerate(chunks[START_INDEX-1:], start=START_INDEX):
        if generated >= N_TO_GENERATE:
            break
        if not snippet.strip():
            continue

        resolved = resolve_references(snippet, references)

        try:
            tf = generate_true_false(resolved, case_name)

            # Robust-ish parsing
            m_stmt = re.search(r"#\s*Statement\s*(.+?)\s*#\s*Answer", tf, re.DOTALL | re.IGNORECASE)
            m_ans  = re.search(r"#\s*Answer\s*(.+?)\s*#\s*Explanation", tf, re.DOTALL | re.IGNORECASE)
            m_expl = re.search(r"#\s*Explanation\s*(.+)$", tf, re.DOTALL | re.IGNORECASE)

            if not (m_stmt and m_ans and m_expl):
                print(f"[Skipped chunk {i}]: Invalid format")
                continue

            statement   = m_stmt.group(1).strip()
            answer_raw  = m_ans.group(1).strip()
            # normalise answer to 'True' or 'False'
            answer      = "True" if answer_raw.lower().startswith("t") else "False" if answer_raw.lower().startswith("f") else answer_raw
            explanation = m_expl.group(1).strip()

            if answer not in ("True", "False"):
                print(f"[Skipped chunk {i}]: Answer not True/False => {answer_raw}")
                continue

            ws.append([
                i,
                xl_clip(xlsx_safe(statement)),
                answer,
                xl_clip(xlsx_safe(explanation))
            ])
            generated += 1

        except Exception as e:
            print(f"[Error at chunk {i}]: {e}")

        time.sleep(SLEEP_SECS)

    # Save to Desktop
    safe_name = sanitize_filename(case_name)
    filename = f"{safe_name}_true_false_results.xlsx"
    out_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
    wb.save(out_path)
    print(f"\nTrue/False results saved to: {out_path}")
