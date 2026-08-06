"""
Test the complete Corrective RAG LangGraph workflow.
"""

from src.graph import corrective_rag_app
from src.graph_state import GraphState


print("\n" + "=" * 60)
print("TESTING COMPLETE CORRECTIVE RAG GRAPH")
print("=" * 60)


# ---------------------------------------------------------
# Create the initial state
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


print("\nSTATE BEFORE GRAPH")
print("-" * 60)

print(
    f"Original question: "
    f"{initial_state['original_question']}"
)

print(
    f"Search question: "
    f"{initial_state['search_question']}"
)

print(
    f"Documents: "
    f"{len(initial_state['documents'])}"
)

print(
    f"Filtered documents: "
    f"{len(initial_state['filtered_documents'])}"
)

print(
    f"Web search needed: "
    f"{initial_state['web_search_needed']}"
)

print(
    f"Answer: "
    f"{initial_state['answer']}"
)


# ---------------------------------------------------------
# Run the complete graph
# ---------------------------------------------------------

final_state = corrective_rag_app.invoke(
    initial_state
)


# ---------------------------------------------------------
# Print the final state
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL STATE AFTER GRAPH")
print("=" * 60)

print(
    f"Original question: "
    f"{final_state['original_question']}"
)

print(
    f"Final search question: "
    f"{final_state['search_question']}"
)

print(
    f"Retrieved documents: "
    f"{len(final_state['documents'])}"
)

print(
    f"Relevant and web documents: "
    f"{len(final_state['filtered_documents'])}"
)

print(
    f"Web search needed: "
    f"{final_state['web_search_needed']}"
)

print(
    f"\nFinal answer:\n"
    f"{final_state['answer']}"
)