from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_documents(urls: list[str]):
    """Load documents from a list of URLs."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs


def chunk_documents(docs):
    """Split documents into chunks using recursive character splitting."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    splits = text_splitter.split_documents(docs)
    return splits