"""
Vectore store configuration for Corrective RAG project. 
This file:
1- load the sample documents.
2- Create Bedrock embeddings.
3- Store the documents in a vector Chroma. 
4- Create a retriever for document search

"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.llm import get_embedding_model
from src.sample_data import sample_documents


def create_vector_store() -> Chroma:
    """
    Create an in-memory Chroma vectore store.

    Each sample document is converted into an embedding and stored in Chroma. 
    """
    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=sample_documents,
        embedding=embedding_model,
        collection_name="corrective_rag"

    )
    return vector_store

def create_retriever(
        vector_store = Chroma, 
        number_of_documents: int = 3,
        search_type: str = "similarity"      
) -> VectorStoreRetriever:

    """
    Create a retriever from the chroma vector store.

    Args:
    vector_store (Chroma): The Chroma vector store instance.
    number_of_documents (int): The number of documents to retrieve.
    search_type (str): The type of search to perform, e.g., "similarity".

    Returns:
    VectorStoreRetriever: A retriever for the Chroma vector store that performs similarity search.
    """
    retriver = vector_store.as_retriever(
    search_kwargs={
        "k": number_of_documents,
       }
    )
    return retriver


