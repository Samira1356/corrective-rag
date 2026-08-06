"""
Shared state for the Corrective RAG workflow.

The state carries information from one LangGraph node
to the next node.
"""

# TypedDict lets us describe the expected structure
# of a Python dictionary.
from typing import TypedDict

# Document is a LangChain class used to store:
# - text content
# - metadata about the text
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Defines all information that can move through
    the Corrective RAG workflow.
    """

    # The exact question entered by the user.
    original_question: str

    # The question currently used for retrieval.
    #
    # At the beginning, this will be the same as
    # original_question.
    #
    # Later, the LLM may rewrite it for web search.
    search_question: str

    # Documents retrieved from the local vector store.
    documents: list[Document]

    # Documents that the grader decides are relevant.
    filtered_documents: list[Document]

    # True means the local documents are not sufficient
    # and the workflow should perform a web search.
    web_search_needed: bool

    # The final answer generated for the user.
    answer: str