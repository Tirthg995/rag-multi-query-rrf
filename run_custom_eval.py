import os
os.environ["USER_AGENT"] = "rag-from-scratch-project"

from src.ingestion import load_documents, chunk_documents
from src.embeddings import get_embedding_model, build_vectorstore, get_retriever
from src.llm import get_llm
from src.multi_query import build_multi_query_chain
from src.rerank import build_rrf_retrieval_chain
from src.generation import build_rag_chain
from src.baseline import build_naive_rag_chain
from src.eval_data import EVAL_QUESTIONS
from src.custom_eval import evaluate_pipeline, summarize_results, jaccard_similarity


def main():
    # ── Setup ──
    urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
    docs = load_documents(urls)
    splits = chunk_documents(docs)

    embedding_model = get_embedding_model()
    vectorstore = build_vectorstore(splits, embedding_model)
    retriever = get_retriever(vectorstore)

    query_llm = get_llm(decoding_method="sample", temperature=0.7)
    generate_queries = build_multi_query_chain(query_llm)
    retrieval_chain_rrf = build_rrf_retrieval_chain(generate_queries, retriever)

    answer_llm = get_llm(decoding_method="greedy", temperature=0.0)
    rag_chain = build_rag_chain(retrieval_chain_rrf, answer_llm)
    naive_chain = build_naive_rag_chain(retriever, answer_llm, top_k=5)

    print(f"Running custom eval on {len(EVAL_QUESTIONS)} questions...\n")

    # ── Evaluate both pipelines ──
    print("Evaluating full pipeline (Multi-Query + RRF)...")
    full_results = evaluate_pipeline(rag_chain, retrieval_chain_rrf, EVAL_QUESTIONS, is_naive=False)

    print("Evaluating naive baseline (single query, top-k)...\n")
    naive_results = evaluate_pipeline(naive_chain, retriever, EVAL_QUESTIONS, is_naive=True)

    # ── Summarize ──
    full_summary = summarize_results(full_results)
    naive_summary = summarize_results(naive_results)

    # ── Print comparison table ──
    print("=" * 60)
    print("CUSTOM EVAL RESULTS COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Full Pipeline':<15} {'Naive Baseline':<15}")
    print("-" * 60)
    for metric in ["context_coverage", "chunk_precision", "answer_coverage"]:
        full_score = full_summary[metric]
        naive_score = naive_summary[metric]
        print(f"{metric:<25} {full_score:<15.4f} {naive_score:<15.4f}")

    # ── Average Jaccard divergence between retrieval sets ──
    jaccard_scores = [
        jaccard_similarity(f["contexts"], n["contexts"])
        for f, n in zip(full_results, naive_results)
    ]
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    print(f"\nAvg. retrieval set overlap (Jaccard) between pipelines: {avg_jaccard:.4f}")
    print("(Lower = multi-query is retrieving meaningfully different chunks than naive search)")

    # ── Per-question breakdown (useful for spotting where each pipeline struggles) ──
    print("\n" + "=" * 60)
    print("PER-QUESTION BREAKDOWN")
    print("=" * 60)
    for f, n in zip(full_results, naive_results):
        print(f"\nQ: {f['question']}")
        print(f"  Full  -> coverage: {f['context_coverage']:.2f}  precision: {f['chunk_precision']:.2f}  answer: {f['answer_coverage']:.2f}")
        print(f"  Naive -> coverage: {n['context_coverage']:.2f}  precision: {n['chunk_precision']:.2f}  answer: {n['answer_coverage']:.2f}")


if __name__ == "__main__":
    main()