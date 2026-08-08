# RAG From Scratch: Multi-Query + Reciprocal Rank Fusion

A Retrieval-Augmented Generation (RAG) pipeline built from first principles — custom document chunking, multi-query translation, and Reciprocal Rank Fusion (RRF) — benchmarked against a naive top-k retrieval baseline using a custom deterministic evaluation harness.

## Overview

Standard RAG retrieves documents using a single query embedding, which is sensitive to how the question happens to be phrased. This project implements **multi-query translation** (generating 5 rephrasings of each question) combined with **Reciprocal Rank Fusion** to merge multiple ranked retrieval lists into one, aiming to improve retrieval recall over naive single-query search.

The full pipeline is benchmarked against a naive baseline (single query, top-k retrieval) on 11 questions spanning both simple lookups and complex, multi-faceted queries.

## Architecture

Documents → Chunking → Embedding → Vectorstore
│
Question → Multi-Query (5 variants) → Retrieve (×5) → RRF Fusion → Top-K Context → LLM → Answer
│
[Naive baseline skips multi-query/RRF,
retrieves directly on original question]


## Stack
- **LLM**: Groq (`llama-3.1-8b-instant` for pipeline generation, `llama-3.3-70b-versatile` reserved where higher reasoning quality matters)
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vectorstore**: Chroma (local)
- **Orchestration**: LangChain (LCEL)
- **Evaluation**: Custom deterministic keyword-overlap harness (see [Evaluation Methodology](#evaluation-methodology))

## Results

Evaluated on 11 questions (6 direct-lookup, 5 complex/multi-faceted) against Lilian Weng's "LLM Powered Autonomous Agents" blog post.

| Metric | Full Pipeline (Multi-Query + RRF) | Naive Baseline (Top-K) |
|---|---|---|
| Context Coverage (recall proxy) | **0.58** | 0.38 |
| Chunk Relevance Precision | 0.83 | **0.95** |
| Answer Keyword Coverage | **0.49** | 0.40 |

**Key finding:** Multi-query + RRF trades ~13% retrieval precision for ~53% higher context coverage, yielding measurably more complete answers (+21% answer coverage) — with the advantage most pronounced on broad, open-ended questions (e.g. "What types of memory does an LLM agent use?": 0.58 vs 0.19 context coverage). On narrow, single-concept questions, the naive baseline's tighter precision performs comparably or better, since there's less to gain from exploring multiple query angles.

This reflects a real precision/recall trade-off rather than a uniform win — multi-query retrieval is most valuable when a question can't be answered by a single well-matched chunk.

## Evaluation Methodology

Initially built on RAGAS (LLM-as-judge scoring), but repeatedly hit free-tier token rate limits (watsonx, then Groq) that made reliable evaluation runs impractical. Replaced with a custom, fully deterministic evaluation harness:

- **Context Coverage** — fraction of ground-truth keywords found anywhere in retrieved chunks (recall proxy)
- **Chunk Relevance Precision** — fraction of retrieved chunks containing at least one ground-truth keyword (precision proxy)
- **Answer Keyword Coverage** — fraction of ground-truth keywords present in the generated answer
- **Jaccard Divergence** — set overlap between the two pipelines' retrieved chunks, quantifying how much multi-query actually changes retrieval

No LLM API calls are required for scoring — pure keyword/set operations — making the evaluation fast, free, and reproducible.

## Project Structure

src/
├── config.py # env vars, constants
├── ingestion.py # document loading + chunking
├── embeddings.py # embedding model + vectorstore
├── llm.py # LLM provider connection
├── multi_query.py # multi-query generation chain
├── rerank.py # Reciprocal Rank Fusion
├── generation.py # final RAG answer chain
├── baseline.py # naive top-k retrieval (control group)
├── eval_data.py # hand-written eval question set
└── custom_eval.py # deterministic evaluation metrics
main.py # demo: runs one question through both pipelines
run_custom_eval.py # benchmark: runs full eval set, prints comparison


## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

GROQ_API_KEY=your_key_here


## Usage

```bash
python main.py              # run pipeline on a test question
python run_custom_eval.py    # run full benchmark comparison
```

## What I'd improve next
- Expand the eval set beyond 11 questions for more statistically stable results
- Test on a messier, real-world corpus (PDFs, multi-document) rather than a single blog post
- Explore late chunking and contextual retrieval as alternative indexing strategies (Phase 3 of this project's broader roadmap)