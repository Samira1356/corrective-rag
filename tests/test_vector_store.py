"""
Test the Chroma vectore store and retriever.
"""

from src.vector_store import create_vector_store, create_retriever

print("\n" + "=" * 60)
print("TESTING CHROMA VECTOR STORE")
print("=" * 60)

# Create the vector store.
vector_store = create_vector_store()
print(f"Vector store type: {type(vector_store).__name__}")

# Create the retriever.
retriever = create_retriever(vector_store=vector_store, number_of_documents=3, search_type="similarity")
print(f"Retriever type: {type(retriever).__name__}")

# Search the vector store. 
question = "How does Corrective RAG handle irrelevant documents?"

retrieved_documents = retriever.invoke(question)

print("\n" + "=" * 60)
print("RETRIEVAL QUESTION")
print("=" * 60)

print(question)

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)
print(f"Number of documents retrieved: {len(retrieved_documents)}")


for number, document in enumerate(retrieved_documents, start=1):
    print("\n" + "-" * 60)
    print(f"DOCUMENT {number}")
    print("-" * 60)
    print(f"Content: {document.page_content}")
    print(f"Metadata: {document.metadata}")