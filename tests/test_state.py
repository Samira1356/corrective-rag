"""
Test the GraphState structure.

This is a simple Python test.
We are not building the LangGraph workflow yet.
"""

# Import GraphState from the file we created.
from src.graph_state import GraphState


# Create an actual dictionary that follows
# the GraphState structure.
initial_state: GraphState = {
    "original_question": "What is Corrective RAG?",
    "search_question": "What is Corrective RAG?",
    "documents": [],
    "filtered_documents": [],
    "web_search_needed": False,
    "answer": "",
}


print("\n" + "=" * 60)
print("INITIAL GRAPH STATE")
print("=" * 60)

print(f"Original question: {initial_state['original_question']}")
print(f"Search question: {initial_state['search_question']}")
print(f"Documents: {initial_state['documents']}")
print(f"Filtered documents: {initial_state['filtered_documents']}")
print(f"Web search needed: {initial_state['web_search_needed']}")
print(f"Answer: {initial_state['answer']}")