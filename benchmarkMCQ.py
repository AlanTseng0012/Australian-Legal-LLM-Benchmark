import fitz
import re
import time
import os
import string
from openai import OpenAI
from semchunk import chunk_raw
from tiktoken import get_encoding
from functools import lru_cache
from openpyxl import Workbook
from dotenv import load_dotenv, find_dotenv

# Setup OpenAI client
load_dotenv(find_dotenv())
client = OpenAI()
_tokenizer = get_encoding("cl100k_base")

@lru_cache(maxsize=2048)
def token_counter(text: str) -> int:
    return len(_tokenizer.encode(text))

def load_semantic_chunks(file_path, max_tokens=1024, print_chunks=False):
    full_text, spans = build_full_text_and_spans(file_path)

    chunks = chunk_raw(full_text, token_counter=token_counter, max_tokens=max_tokens)

    # Find each chunk's absolute start index in full_text (scan forward)
    chunk_with_meta = []
    search_from = 0
    for c in chunks:
        idx = full_text.find(c, search_from)
        if idx == -1:  # fallback: brute search from 0 if needed
            idx = full_text.find(c)
        if idx == -1:
            # skip if we somehow can't locate it
            continue
        start = idx
        end = idx + len(c)
        # Which pages intersect this chunk?
        pages_here = [pno for (s,e,pno) in spans if not (end <= s or start >= e)]
        chunk_with_meta.append((c, pages_here, start, end))
        search_from = end

    filtered = [(c, pgs, start, end) for (c, pgs, start, end) in chunk_with_meta
                if len(c.split()) > 20 and "high court" not in c.lower()]

    if print_chunks:
        for i, (c, pgs, _, __) in enumerate(filtered, 1):
            print(f"\n--- Chunk {i} (pages {pgs}) ---")
            print(c)
            print("-" * 80)

    return filtered, spans, full_text

def safe_get_text(page):
    for mode in ("text", None):  # None => default
        try:
            return page.get_text(mode) if mode else page.get_text()
        except Exception:
            pass
    return page.get_text()  # last resort

def build_full_text_and_spans(file_path):
    doc = fitz.open(file_path)
    pages, spans, cursor = [], [], 0
    for pno, page in enumerate(doc, start=1):
        t = safe_get_text(page)
        pages.append(t)
        start, end = cursor, cursor + len(t)
        spans.append((start, end, pno))
        cursor = end + 1
    full_text = "\n".join(pages)
    return full_text, spans

def extract_case_name(file_path):
    doc = fitz.open(file_path)
    first_page_text = doc[0].get_text()
    lines = first_page_text.splitlines()
    combined_text = " ".join(lines)

    match_case = re.search(r'([A-Z][A-Za-z&()\s.,]+? v [A-Z][A-Za-z&()\s.,]+?)\s*\[(\d{4})\]\s*HCA\s*(\d{1,3})', combined_text, flags=re.IGNORECASE)
    if match_case:
        case_name = match_case.group(1).strip()
        year = match_case.group(2)
        number = match_case.group(3)
        return f"{case_name} [{year}] HCA {number}"

    for line in lines:
        match_act = re.search(r'([A-Za-z\s]+Act\s+(?:19|20)\d{2})', line)
        if match_act:
            return match_act.group(1).strip()

    return "an Australian legal document"

def extract_references_from_pdf(file_path):
    doc = fitz.open(file_path)
    lines = [line.strip() for page in doc for line in page.get_text().split('\n') if line.strip()]
    reference_dict = {}
    current_key = None
    for line in lines:
        match = re.match(r"\[(\d{1,3})\]\s*(.+)", line)
        if match:
            ref_num = int(match.group(1))
            if len(str(ref_num)) == 4:
                continue
            current_key = ref_num
            reference_dict[current_key] = match.group(2).strip()
        elif current_key and line:
            reference_dict[current_key] += " " + line.strip()
    return dict(sorted(reference_dict.items()))

def resolve_references(snippet, reference_dict):
    for ref, citation in reference_dict.items():
        snippet = snippet.replace(f"[{ref}]", f"{citation} [{ref}]")
    return snippet

def sanitize_filename(name, fallback="Australian_legal_document", max_length=100):
    safe_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    cleaned = ''.join(c for c in name if c in safe_chars).strip()
    if len(cleaned) > max_length:
        return fallback
    return cleaned or fallback

def extract_support_and_page(resolved_chunk, options, correct_letters, full_text, spans, chunk_start):
    # Build target text from correct options
    targets = [options.get(k, "") for k in correct_letters]
    targets_text = " ".join(targets)

    # Split chunk into sentences (very simple split)
    sentences = re.split(r'(?<=[\.\?\!])\s+', resolved_chunk.strip())
    if not sentences:
        return "", None

    # Score sentences by word overlap with targets (bag-of-words)
    tgt_words = set(w.lower() for w in re.findall(r'\w+', targets_text))
    scored = []
    cursor = 0
    for s in sentences:
        s_words = set(w.lower() for w in re.findall(r'\w+', s))
        score = len(tgt_words & s_words)
        # find sentence offset within full_text by searching from chunk_start + cursor
        local_idx = resolved_chunk.find(s, cursor)
        if local_idx == -1:
            local_idx = resolved_chunk.find(s)  # fallback
        abs_start = chunk_start + max(local_idx, 0)
        scored.append((score, s, abs_start))
        cursor = (local_idx + len(s)) if local_idx != -1 else cursor

    # pick top 1-2 sentences with score > 0, else fallback first sentence
    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [s for sc, s, _ in scored if sc > 0][:2] or [scored[0][1]]

    # page = page covering the first picked sentence
    first_abs = next((abs_start for sc, s, abs_start in scored if s == picks[0]), None)
    page = None
    if first_abs is not None:
        for (s,e,pno) in spans:
            if s <= first_abs < e:
                page = pno
                break

    # Return a compact quote
    quote = " ".join(picks)
    return quote, page

def generate_mcq(snippet, case_name):
    case_context = f"This text is from the case *{case_name}*.\n\n"
    prompt = f"""{case_context}Based on the snippet below, generate ONE multiple-choice question with exactly 4 options.

<snippet>
{snippet}
</snippet>

# Format
You must format your response as follows:
<format>
# Question
{{The question here}}

# Options
(A) {{option A}}
(B) {{option B}}
(C) {{option C}}
(D) {{option D}}

# Correct Answer
{{A/B/C/D}}
</format>

# Instructions
- Your question must be decontextualised, standalone, and must reference the full case name exactly as provided above.
- There should be more than two correct answers. Include all correct options and clearly supported by the snippet.
- The incorrect answers should be plausible but not supported by the text.
- Do not use "All of the above" or "None of the above" as options.
"""

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0
    )
    return completion.choices[0].message.content.strip()

# --- MAIN ---
START_INDEX = 11
N_TO_GENERATE = 3

if __name__ == "__main__":
    file_path = input("Enter path to your legal PDF file: ").strip()
    case_name = extract_case_name(file_path)
    references = extract_references_from_pdf(file_path)
    chunks_meta, spans, full_text = load_semantic_chunks(file_path)

    wb = Workbook()
    ws = wb.active
    ws.title = "MCQ"
    ws.append(["ChunkIndex","Question","Option A","Option B","Option C","Option D","Correct Answer","Page","Explanation"])

    count = 0
    # iterate starting at the START_INDEX chunk
    for i, (snippet, pages, chunk_start, chunk_end) in enumerate(chunks_meta[START_INDEX:], start=START_INDEX):
        if count >= N_TO_GENERATE:
            break
        if not snippet.strip():
            continue

        print(f"\n--- Chunk {i+1} ---")
        resolved = resolve_references(snippet, references)
        try:
            mcq = generate_mcq(resolved, case_name)

            if "# Question" in mcq and "# Options" in mcq and "# Correct Answer" in mcq:
                question_match = re.search(r"# Question\n(.+?)\n# Options", mcq, re.DOTALL)
                options_block_match = re.search(r"# Options\n(.+?)\n# Correct Answer", mcq, re.DOTALL)
                correct_block_match = re.search(r"# Correct Answer\n(.+)", mcq, re.DOTALL)

                if not (question_match and options_block_match and correct_block_match):
                    print(f"[Skipped chunk {i+1}]: Invalid format")
                    continue

                question = question_match.group(1).strip()
                options_block = options_block_match.group(1).strip()

                # Extract just the letters in correct answer block
                correct_block = correct_block_match.group(1).strip()
                letters = re.findall(r'\b([A-D])\b', correct_block)
                order = ['A','B','C','D']
                correct = ", ".join([x for x in order if x in dict.fromkeys(letters)])  # keeps original order

                # Parse options
                options = dict(re.findall(r"[\(]?([A-D])[\)]?\s*[\:\.]?\s+(.+)", options_block))

                # parse letters in order A-D
                correct_letters = [x.strip() for x in correct.split(",") if x.strip() in ("A","B","C","D")]

                # minimal support + the single page where that support appears
                support_text, answer_page = extract_support_and_page(
                    resolved, options, correct_letters, full_text, spans, chunk_start
                )

                # sanitize for Excel
                _ILLEGAL_XLSX = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
                def xlsx_safe(s: str) -> str:
                    return _ILLEGAL_XLSX.sub("", s) if isinstance(s, str) else s

                ws.append([
                    i + 1,
                    xlsx_safe(question),
                    xlsx_safe(options.get("A","")),
                    xlsx_safe(options.get("B","")),
                    xlsx_safe(options.get("C","")),
                    xlsx_safe(options.get("D","")),
                    xlsx_safe(correct),
                    xlsx_safe(str(answer_page or "")),
                    xlsx_safe(support_text)
                ])
                count += 1

        except Exception as e:
            print(f"[Error at chunk {i+1}]: {e}")
        time.sleep(6)

    safe_name = sanitize_filename(case_name)
    filename = f"{safe_name}_mcq_results.xlsx"
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", filename)
    wb.save(desktop_path)
    print(f"\n==== Q&A results saved to: {desktop_path} ====")

