def reciprocal_rank_fusion(results: list[list], k: int = 60):
    """
    Fuse multiple ranked lists of documents into a single ranked list
    using Reciprocal Rank Fusion (RRF).

    Each document's fused score is the sum of 1/(rank + k) across every
    list it appears in — so a document ranked highly in multiple lists
    outranks one that only appears once, even at rank 1.
    """
    fused_scores = {}
    doc_lookup = {}

    for docs_list in results:
        for rank, doc in enumerate(docs_list):
            doc_key = doc.page_content
            doc_lookup[doc_key] = doc
            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0
            fused_scores[doc_key] += 1 / (rank + k)

    reranked_results = [
        (doc_lookup[doc_key], score)
        for doc_key, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def build_rrf_retrieval_chain(generate_queries_chain, retriever):
    """
    Build the full chain: generate multiple queries -> retrieve for each -> fuse with RRF.
    """
    return generate_queries_chain | retriever.map() | reciprocal_rank_fusion