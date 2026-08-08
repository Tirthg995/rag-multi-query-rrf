from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY

def get_llm(decoding_method="sample", temperature=0.7, max_tokens=300, model="llama-3.1-8b-instant"):
    """
    Return a configured ChatGroq instance running Llama 3.3 70B.
    decoding_method is kept as a parameter for interface compatibility with
    the rest of the pipeline, but Groq controls variation via temperature only.
    """
    return ChatGroq(
        model=model,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )