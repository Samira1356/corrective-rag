"""
Load and split PDF documents for the Corrective RAG project.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):
    """
    Load a PDF and return LangChain Document objects.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents


def split_documents(documents):
    """
    Split loaded documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks