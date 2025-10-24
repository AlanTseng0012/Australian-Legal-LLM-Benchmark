# Australian-Legal-LLM-Benchmark
This repository provides the experimental pipeline for benchmarking **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** systems in the context of **Australian contract law**.

## 📘 Overview
The benchmark evaluates both **LLM-only (closed-book)** and **RAG (document-grounded)** configurations across multiple legal reasoning tasks — Multiple Choice (MCQ), True/False (TF), and open-ended Question Answering (QA).  
It uses curated datasets derived from **High Court of Australia cases** and **key statutory sources**.

## 🧩 Key Components
LANGCHAIN_RAG/
| File / Folder | Description |
|----------------|-------------|
| **benchmark_data/** | Contains curated benchmark datasets and evaluation results. |
| **Cases processing/** | Scripts and notebooks for raw legal PDF preprocessing. |
| **faiss_index/** | Auto-generated FAISS index storage for vector embeddings. |
| **benchmarkMCQ.py / benchmarkTF.py / benchmarkQA.py** | Generate different benchmark question formats (MCQ, True/False, QA). |
| **ingest.py** | Embeds legal documents and builds the FAISS index. |
| **query.py** | Handles RAG queries and retrieval logic. |
| **footnote_loader.py** | Custom loader to preserve legal document footnotes. |
| **llmOnly.py** | Closed-book (no retrieval) model benchmarking. |
| **llmAsAJudge.py** | Evaluates model-generated answers using an LLM-as-judge approach. |
| **requirements.txt** | List of Python dependencies required to run the project. |
| **.env.example** | Template for environment variables (e.g., API keys). |


## 🧠 Key Features
- Modular pipeline supporting both **OpenAI** and **domain-specific embeddings** (`adlumal/auslaw-embed-v1.0`)
- Semantic chunking with legal footnote retention
- RAG and zero-shot baselines for comparison
- Reproducible evaluation scripts (for each question type)

## 📊 Legal Data
- **High Court Cases:** 7 major contract law precedents
- **Statutes:** 3 key Acts (Insurance Contracts Act 1984, Independent Contractors Act 2006, Contracts Review Act 1980)

## 🚀 Quick Start
```bash
# Clone the repository
git clone https://github.com/<your-username>/auslaw-benchmark-rag.git
cd auslaw-benchmark-rag

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
