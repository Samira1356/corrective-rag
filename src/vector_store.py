"""
Crate the chroma vector store and retriever for the CAQH Index 2024 PDF.
"""
from langchain_chroma import Chroma

from src.document_loader import load_pdf, split_documents
from src.llm import get_embedding_model

PDF_PATH = "data/caqh_index_2024.pdf"

def create_vector_store():
    documents = load_pdf(PDF_PATH)
    chunks = split_documents(documents) 
    embedding_model = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="caqh_index_2024",  
    )
    return vector_store

# the number of documents retrieved was updated from 3 to 5 due to better evaluation. 
def create_retriever(vector_store, number_of_documents=5):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": number_of_documents}
    )
    return retriever    