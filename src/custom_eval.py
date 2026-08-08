import re

# Common English stopwords to filter out when extracting keywords
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "that", "this", "these", "those",
    "it", "its", "their", "they", "them", "which", "who", "what", "how",
    "can", "could", "may", "might", "will", "would", "should", "does",
    "do", "did", "has", "have", "had", "not", "no", "such", "into",
    "over", "also", "than", "when", "where", "each", "other", "some",
    "used", "using", "use", "based", "including", "e.g",
}


def extract_keywords(text: str, min_length: int = 4) -> set:
    """
    Extract meaningful keywords from a piece of text: lowercase,
    strip punctuation, filter stopwords and short words.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if len(w) >= min_length and w not in STOPWORDS}


def context_keyword_coverage(ground_truth: str, retrieved_chunks: list[str]) -> float:
    """
    Of the ground-truth keywords, what fraction appear somewhere in the
    retrieved context? Proxy for retrieval RECALL.
    """
    gt_keywords = extract_keywords(ground_truth)
    if not gt_keywords:
        return 0.0

    combined_context = " ".join(retrieved_chunks).lower()
    context_words = extract_keywords(combined_context)

    found = gt_keywords & context_words
    return len(found) / len(gt_keywords)


def chunk_relevance_precision(ground_truth: str, retrieved_chunks: list[str]) -> float:
    """
    Of the retrieved chunks, what fraction contain at least one
    ground-truth keyword? Proxy for retrieval PRECISION.
    """
    gt_keywords = extract_keywords(ground_truth)
    if not gt_keywords or not retrieved_chunks:
        return 0.0

    relevant_count = 0
    for chunk in retrieved_chunks:
        chunk_words = extract_keywords(chunk)
        if gt_keywords & chunk_words:
            relevant_count += 1

    return relevant_count / len(retrieved_chunks)


def answer_keyword_coverage(ground_truth: str, generated_answer: str) -> float:
    """
    Of the ground-truth keywords, what fraction appear in the generated
    answer? Proxy for answer correctness/faithfulness.
    """
    gt_keywords = extract_keywords(ground_truth)
    if not gt_keywords:
        return 0.0

    answer_words = extract_keywords(generated_answer)
    found = gt_keywords & answer_words
    return len(found) / len(gt_keywords)


def jaccard_similarity(chunks_a: list[str], chunks_b: list[str]) -> float:
    """
    Jaccard similarity between two sets of retrieved chunks (by content).
    Quantifies how different two retrieval strategies' results actually are.
    """
    set_a = set(chunks_a)
    set_b = set(chunks_b)
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def evaluate_pipeline(chain, retrieval_chain, questions: list[dict], is_naive: bool = False):
    """
    Run a pipeline over the eval questions and compute all custom metrics.
    Returns a list of per-question results plus the raw retrieved chunks
    (needed later for Jaccard comparison between pipelines).
    """
    results = []

    for item in questions:
        q = item["question"]
        gt = item["ground_truth"]

        answer = chain.invoke({"question": q})
        if hasattr(answer, "content"):  # handle ChatGroq message objects
            answer = answer.content

        if is_naive:
            retrieved_docs = retrieval_chain.invoke(q)
            contexts = [doc.page_content for doc in retrieved_docs]
        else:
            fused_results = retrieval_chain.invoke({"question": q})
            contexts = [doc.page_content for doc, score in fused_results]

        results.append({
            "question": q,
            "ground_truth": gt,
            "answer": answer,
            "contexts": contexts,
            "context_coverage": context_keyword_coverage(gt, contexts),
            "chunk_precision": chunk_relevance_precision(gt, contexts),
            "answer_coverage": answer_keyword_coverage(gt, answer),
        })

    return results


def summarize_results(results: list[dict]) -> dict:
    """Average the per-question metrics into overall scores."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "context_coverage": sum(r["context_coverage"] for r in results) / n,
        "chunk_precision": sum(r["chunk_precision"] for r in results) / n,
        "answer_coverage": sum(r["answer_coverage"] for r in results) / n,
    }