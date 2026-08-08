from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

RAG_TEMPLATE = """Answer the following question based on this context:

{context}

Question: {question}
"""


def format_docs(fused_results, top_n: int = 8):
    """Take top N fused (doc, score) tuples and join their content into one context string."""
    top_docs = [doc for doc, score in fused_results[:top_n]]
    return "\n\n".join(doc.page_content for doc in top_docs)


def build_rag_chain(retrieval_chain, llm):
    """
    Build the full RAG chain: retrieve+fuse -> format context -> generate answer.
    retrieval_chain should be the RRF chain from src.rerank.build_rrf_retrieval_chain.
    """
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    chain = (
        {
            "context": retrieval_chain | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain