"""
Test the first two Corrective RAG node functions.

This test runs:

1. retrieve_documents()
2. grade_documents()

We are calling the node functions manually.
LangGraph is not connected yet.
"""

from src.graph_state import GraphState
from src.nodes import grade_documents, retrieve_documents


print("\n" + "=" * 60)
print("TESTING CORRECTIVE RAG NODES")
print("=" * 60)


# ---------------------------------------------------------
# Create the initial graph state
# ---------------------------------------------------------

initial_state: GraphState = {
    "original_question": (
        "How does Corrective RAG handle "
        "insufficient retrieved information?"
    ),
    "search_question": (
        "How does Corrective RAG handle "
        "insufficient retrieved information?"
    ),
    "documents": [],
    "filtered_documents": [],
    "web_search_needed": False,
    "answer": "",
}


print("\nINITIAL STATE")
print("-" * 60)

print(f"Original question: {initial_state['original_question']}")
print(f"Documents: {initial_state['documents']}")
print(
    f"Filtered documents: "
    f"{initial_state['filtered_documents']}"
)


# ---------------------------------------------------------
# Run the retrieval node
# ---------------------------------------------------------

retrieval_update = retrieve_documents(initial_state)


# Merge the returned update into the state.
initial_state.update(retrieval_update)


print("\nSTATE AFTER RETRIEVAL")
print("-" * 60)

print(
    f"Documents retrieved: "
    f"{len(initial_state['documents'])}"
)


for number, document in enumerate(
    initial_state["documents"],
    start=1,
):
    print(
        f"\nDocument {number}: "
        f"{document.page_content}"
    )


# ---------------------------------------------------------
# Run the grading node
# ---------------------------------------------------------

grading_update = grade_documents(initial_state)


# Merge the grading update into the state.
initial_state.update(grading_update)


print("\nFINAL STATE AFTER GRADING")
print("-" * 60)

print(
    f"Relevant documents: "
    f"{len(initial_state['filtered_documents'])}"
)

print(
    f"Web search needed: "
    f"{initial_state['web_search_needed']}"
)


for number, document in enumerate(
    initial_state["filtered_documents"],
    start=1,
):
    print(
        f"\nRelevant document {number}: "
        f"{document.page_content}"
    )