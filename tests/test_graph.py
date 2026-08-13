"""
Test the complete Corrective RAG LangGraph workflow.
using the CAQH pdf healthcare data.
"""

from src.graph import corrective_rag_app
from src.graph_state import GraphState

question = (
    "How can automation reduce administrative burden "
    "in healthcare?"
)

initial_state: GraphState = {
    "original_question": question,
    "search_question": question,
    "documents": [],
    "filtered_documents": [],
    "web_search_needed": False,
    "answer": "",
}


final_state = corrective_rag_app.invoke(
    initial_state
)

print("Original question:", 
      final_state["original_question"])

print(
    "Retrieved documents:",
    len(final_state["documents"]),
)

print(
    "Relevant documents:",
    len(final_state["filtered_documents"]),
)

print(
    "Web search needed:",
    final_state["web_search_needed"],
)

print(
    "Final answer:",
    final_state["answer"],
)
