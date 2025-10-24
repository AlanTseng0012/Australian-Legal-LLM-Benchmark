# query.py
from dotenv import load_dotenv
import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

EMBED_MODEL = "adlumal/auslaw-embed-v1.0"
INDEX_DIR = f"faiss_index"

SYSTEM_PROMPT = """You are a careful legal research assistant.
Answer the user's question using ONLY the provided context from Australian legal PDFs.
- Always provide concise citations as: (Source: <filename>, p.<page>).
- Prefer quoting or paraphrasing the most relevant passages.

If the question is True/False:
1) On the FIRST line, output "TRUE" or "FALSE" in ALL CAPS.
2) On the SECOND line, give a one-sentence justification based on the context.

If the user's question contains multiple-choice options labeled A), B), C), D):
1) On the FIRST line, output ALL correct letters in UPPERCASE, separated by a SINGLE SPACE if you think more than one answers.
2) On the SECOND line, give a one-sentence justification based on the context.

If the question is Question/Answer:
1) On the FIRST line, output the answer in a concise manner.
2) On the SECOND line, give a one-sentence justification based on the context.

Context:
{context}
"""

def format_docs(docs: List[Document]) -> str:
    out = []
    for d in docs:
        meta = d.metadata or {}
        fn = meta.get("source", "unknown.pdf")
        page = meta.get("page_display") or meta.get("page") or "?"
        out.append(f"[{fn} p.{page}] {d.page_content.strip()}")
    return "\n\n".join(out)

def main():
    load_dotenv()
    # ChatOpenAI uses OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in a .env file.")

    print(f"Loading FAISS index from: {INDEX_DIR}")

    # same embedder as ingest.py (384-d, normalized)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # load the matching index
    vectordb = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    print("Index dim:", vectordb.index.d)
    print("Embed dim:", len(embeddings.embed_query("dim-probe")))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, top_p=1)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=SYSTEM_PROMPT + "\nQuestion: {question}\nAnswer:"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context", "verbose": False},
        return_source_documents=True,
    )

    print("Ask legal questions about your PDFs. Type 'exit' to quit.")
    while True:
        q = input("\nQ: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        result = qa.invoke({"query": q})
        answer = result["result"]
        sources: List[Document] = result.get("source_documents", [])

        seen = set()
        cits = []
        for d in sources:
            fn = (d.metadata or {}).get("source", "unknown.pdf")
            page = (d.metadata or {}).get("page_display") or (d.metadata or {}).get("page") or "?"
            key = (fn, page)
            if key not in seen:
                cits.append(f"{fn} p.{page}")
                seen.add(key)

        print("\nA:", answer)
        if cits:
            print("Sources:", "; ".join(cits))

if __name__ == "__main__":
    main()