"""
Test chroma vector store and retriever with PDF documents.
using the CAQH pdf healthcare data.
"""
from src.vector_store import create_vector_store, create_retriever 

vector_store = create_vector_store()

print("vector store type: ", type(vector_store).__name__)

retriever = create_retriever(
    vector_store = vector_store,
    number_of_documents = 3,
                             )

print("retriever type: ", type(retriever).__name__)

question = (
    "What healthcare administrative transactions "
    "have opportunities for automation?"
)

print("Question: ", question)

retrieved_documents = retriever.invoke(question)

print("Number of retrieved documents: ", len(retrieved_documents))

for number, document in enumerate(retrieved_documents, start=1):
    print(f"\nDocument {number}:")
    print(document.page_content)
    print("Metadata: ", document.metadata)