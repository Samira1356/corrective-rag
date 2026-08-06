"""
Test the sample knowledge documents.

This file confirms that:

1. The documents can be imported.
2. Each item is a LangChain Document.
3. page_content and metadata are available.
"""

from src.sample_data import sample_documents


print("\n" + "=" * 60)
print("SAMPLE KNOWLEDGE DOCUMENTS")
print("=" * 60)

print(f"Total documents: {len(sample_documents)}")


for number, document in enumerate(sample_documents, start=1):
    print("\n" + "-" * 60)
    print(f"DOCUMENT {number}")
    print("-" * 60)

    print(f"Content: {document.page_content}")
    print(f"Metadata: {document.metadata}")