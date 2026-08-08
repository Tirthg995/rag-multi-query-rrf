from src.ingestion import load_documents, chunk_documents
from src.embeddings import get_embedding_model, build_vectorstore, get_retriever
from src.llm import get_llm
from src.multi_query import build_multi_query_chain
from src.rerank import build_rrf_retrieval_chain
from src.generation import build_rag_chain
from src.baseline import build_naive_rag_chain

from src.llm import get_llm

llm = get_llm()
response = llm.invoke("Say hello in one sentence.")
print(response.content)

def main():
    # ── 1. Load + chunk documents ──
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]
    docs = load_documents(urls)
    print(f"[1] Loaded {len(docs)} document(s)")

    splits = chunk_documents(docs)
    print(f"[2] Split into {len(splits)} chunks")

    # ── 2. Embed + build retriever ──
    embedding_model = get_embedding_model()
    vectorstore = build_vectorstore(splits, embedding_model)
    retriever = get_retriever(vectorstore)
    print("[3] Vectorstore + retriever ready")

    # ── 3. LLM for multi-query (sampling, for varied rephrasings) ──
    query_llm = get_llm(decoding_method="sample", temperature=0.7)
    generate_queries = build_multi_query_chain(query_llm)
    print("[4] Multi-query chain ready")

    # ── 4. RRF retrieval chain ──
    retrieval_chain_rrf = build_rrf_retrieval_chain(generate_queries, retriever)
    print("[5] RRF retrieval chain ready")

    # ── 5. Final answer generation (greedy, for consistent answers) ──
    answer_llm = get_llm(decoding_method="greedy", temperature=0.0)
    rag_chain = build_rag_chain(retrieval_chain_rrf, answer_llm)
    print("[6] RAG chain ready\n")

    # ── 6. Naive baseline (for comparison) ──
    naive_llm = get_llm(decoding_method="greedy", temperature=0.0)
    naive_chain = build_naive_rag_chain(retriever, naive_llm, top_k=5)
    print("[7] Naive baseline chain ready\n")


    # ── 7. Run a test question through both chains ──
    question = "What is task decomposition for LLM agents?"

    print("=== Full Pipeline (Multi-Query + RRF) ===")
    answer_full = rag_chain.invoke({"question": question})
    print(answer_full)

    print("\n=== Naive Baseline (single query, top-k) ===")
    answer_naive = naive_chain.invoke({"question": question})
    print(answer_naive)


if __name__ == "__main__":
    main()