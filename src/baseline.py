from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_naive_rag_chain(retriever, llm, top_k: int = 5):
    RAG_TEMPLATE = """Answer the following question based on this context:

{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs[:top_k])

    chain = (
        {
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain