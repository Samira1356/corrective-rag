"""
Test the question-rewriting node manually.

LangGraph is not connected yet.
"""

from src.graph_state import GraphState
from src.nodes import rewrite_question


print("\n" + "=" * 60)
print("TESTING QUESTION REWRITING NODE")
print("=" * 60)


# Create a sample graph state.
state: GraphState = {
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
    "web_search_needed": True,
    "answer": "",
}


print("\nSTATE BEFORE REWRITING")
print("-" * 60)

print(f"Original question: {state['original_question']}")
print(f"Search question: {state['search_question']}")


# Run the rewriting node.
rewrite_update = rewrite_question(state)


# Simulate LangGraph updating the shared state.
state.update(rewrite_update)


print("\nSTATE AFTER REWRITING")
print("-" * 60)

print(f"Original question: {state['original_question']}")
print(f"New search question: {state['search_question']}")