"""
Sample knowledge documents for the Corrective RAG project.

Each Document contains:

1. page_content
   The actual text that can be retrieved.

2. metadata
   Extra information describing the document.
"""

from langchain_core.documents import Document


sample_documents = [
    Document(
        page_content=(
            "Corrective RAG checks whether retrieved documents are relevant "
            "before generating a final answer. If the retrieved information "
            "is insufficient, the workflow can perform an external web search."
        ),
        metadata={
            "source": "corrective_rag_notes",
            "topic": "corrective_rag",
        },
    ),

    Document(
        page_content=(
            "LangGraph represents an AI workflow as nodes connected by edges. "
            "Conditional edges can route the workflow to different nodes "
            "depending on the current state."
        ),
        metadata={
            "source": "langgraph_notes",
            "topic": "langgraph",
        },
    ),

    Document(
        page_content=(
            "A vector store saves numerical embeddings of text and retrieves "
            "documents that are semantically similar to a user's question."
        ),
        metadata={
            "source": "vector_store_notes",
            "topic": "vector_store",
        },
    ),

    Document(
        page_content=(
            "Amazon Bedrock provides managed access to foundation models "
            "through AWS services and APIs."
        ),
        metadata={
            "source": "bedrock_notes",
            "topic": "amazon_bedrock",
        },
    ),

    Document(
        page_content=(
            "Document grading evaluates whether a retrieved document contains "
            "information that can help answer the user's question."
        ),
        metadata={
            "source": "grading_notes",
            "topic": "document_grading",
        },
    ),
]
