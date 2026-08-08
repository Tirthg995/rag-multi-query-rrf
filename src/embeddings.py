from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import EMBEDDING_MODEL_NAME, COLLECTION_NAME


def get_embedding_model():
    """Load the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(splits, embedding_model):
    """Build a Chroma vectorstore from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


def get_retriever(vectorstore):
    """Return a retriever from the vectorstore.by default does cosine similarity search"""
    return vectorstore.as_retriever()
    