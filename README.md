# Australian-Legal-LLM-Benchmark
This repository provides the experimental pipeline for benchmarking **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** systems in the context of **Australian contract law**.

## 📘 Overview
The benchmark evaluates both **LLM-only (closed-book)** and **RAG (document-grounded)** configurations across multiple legal reasoning tasks — Multiple Choice (MCQ), True/False (TF), and open-ended Question Answering (QA).  
It uses curated datasets derived from **High Court of Australia cases** and **key statutory sources**.

## ⚙️ Folder Structure
LANGCHAIN_RAG/
│
├── benchmark_data/
│ ├── 300-question.csv # Full benchmark question set (source, question, type, answer)
│ ├── auslaw_1800benchmark_dataset.csv # Model-level evaluation results (1800 scored responses)
│
├── Cases processing/ # Raw legal PDF preprocessing
├── Cases source/ # Reference materials (optional, not for public repo)
├── faiss_index/ # Local FAISS vector storage (auto-generated)
│
├── benchmarkMCQ.py # Generate MCQ items
├── benchmarkTF.py # Generate True/False items
├── benchmarkQA.py # Generate QA items
│
├── ingest.py # Build FAISS index with embeddings
├── query.py # Query RAG pipeline for question answering
├── footnote_loader.py # Custom PDF loader preserving legal footnotes
├── llmOnly.py # Closed-book baseline evaluation
├── llmAsAJudge.py # LLM-as-judge scoring and comparison
│
├── requirements.txt # Python dependencies
└── .env.example # Example environment variables


## 🧠 Key Features
- Modular pipeline supporting both **OpenAI** and **domain-specific embeddings** (`adlumal/auslaw-embed-v1.0`)
- Semantic chunking with legal footnote retention
- RAG and zero-shot baselines for comparison
- Reproducible evaluation scripts (for each question type)

## 📊 Data
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
