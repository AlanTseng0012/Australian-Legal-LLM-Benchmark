import os, hashlib, shutil, tiktoken
from pathlib import Path
from typing import List, Iterable

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from footnote_loader import load_pdfs_footnote_aware

# ---- config ----
PDF_DIR   = "Cases processing"
INDEX_DIR = "faiss_index"
#EMBED_MODEL = "text-embedding-3-large"
EMBED_MODEL = "adlumal/auslaw-embed-v1.0"
CHUNK_TOKENS = 1024
CHUNK_OVERLAP = 150

FAISS_FILE = Path(INDEX_DIR) / "index.faiss"
PKL_FILE   = Path(INDEX_DIR) / "index.pkl"

# ---- batching helpers ----
_tok = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(_tok.encode(text))

def batches_by_tokens(texts, metas, max_tokens=250_000):
    """Yield (texts_batch, metas_batch) keeping sum(tokens) <= max_tokens."""
    buf_t, buf_m, acc = [], [], 0
    for t, m in zip(texts, metas):
        n = token_len(t)
        if n > max_tokens:
            if buf_t:
                yield buf_t, buf_m
                buf_t, buf_m, acc = [], [], 0
            yield [t], [m]
            continue
        if acc + n > max_tokens and buf_t:
            yield buf_t, buf_m
            buf_t, buf_m, acc = [], [], 0
        buf_t.append(t); buf_m.append(m); acc += n
    if buf_t:
        yield buf_t, buf_m

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=False,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def _hash_text(s: str) -> str:
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()

def dedup_chunks(chunks: List[Document]) -> List[Document]:
    seen, out = set(), []
    for d in chunks:
        k = _hash_text(d.page_content)
        if k not in seen:
            seen.add(k); out.append(d)
    return out

def main():
    load_dotenv()

    # Load PDFs
    all_pages = load_pdfs_footnote_aware(PDF_DIR)
    files = sorted({d.metadata.get("source") for d in all_pages})
    print("Found files:", *[f"  - {f}" for f in files], sep="\n")

    # embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # create or load existing index (only if both files exist)
    vectordb = None
    if FAISS_FILE.exists() and PKL_FILE.exists():
        try:
            vectordb = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded existing FAISS index from ./{INDEX_DIR}")
        except Exception as e:
            print(f"[WARN] Failed to load existing index ({e}). Rebuilding from scratch...")
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
            vectordb = None
    else:
        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    GROUP = 5  # process 5 PDFs at a time
    for i in range(0, len(files), GROUP):
        group = files[i:i+GROUP]
        print("\n=== Processing batch ===")
        for f in group: print("  *", f)
        pages = [p for p in all_pages if p.metadata.get("source") in group]

        chunks = split_docs(pages)
        for j, d in enumerate(chunks):
            m = d.metadata or {}
            m["chunk_id"] = f"{m.get('source','unknown')}::p{m.get('page_display','?')}::#{i}-{j}"
            d.metadata = m
        chunks = dedup_chunks(chunks)
        print(f"  -> {len(chunks)} chunks (after dedup)")

        texts = [d.page_content for d in chunks]
        metas = [d.metadata for d in chunks]

        # build or append in token-safe batches
        if vectordb is None:
            init_done = False
            for tb, mb in batches_by_tokens(texts, metas, max_tokens=250_000):
                if not init_done:
                    vectordb = FAISS.from_texts(tb, embeddings, metadatas=mb, normalize_L2=True)
                    init_done = True
                else:
                    vectordb.add_texts(tb, metadatas=mb)
        else:
            for tb, mb in batches_by_tokens(texts, metas, max_tokens=250_000):
                vectordb.add_texts(tb, metadatas=mb)

        vectordb.save_local(INDEX_DIR)
        print(f"  Saved FAISS index to ./{INDEX_DIR}")

    print("\nDone.")

if __name__ == "__main__":
    main()