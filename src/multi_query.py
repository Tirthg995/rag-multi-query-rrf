from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MULTI_QUERY_TEMPLATE = """You are an AI language model assistant. Your task is to generate 
five different versions of the given user question to retrieve relevant documents from a 
vector database. By generating multiple perspectives on the user question, your goal is to 
help the user overcome some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines. Original question: {question}"""


def build_multi_query_chain(llm):
    """Build a chain that generates 5 rephrased versions of a question."""
    prompt = ChatPromptTemplate.from_template(MULTI_QUERY_TEMPLATE)

    chain = (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )
    return chain