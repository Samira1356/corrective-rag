"""
Build and compile the Corrective RAG LangGraph workflow.
"""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.graph_state import GraphState
from src.nodes import (
    generate_answer,
    grade_documents,
    retrieve_documents,
    rewrite_question,
    web_search,
)


# ---------------------------------------------------------
# Conditional routing function
# ---------------------------------------------------------

def route_after_grading(
    state: GraphState,
) -> Literal["rewrite_question", "generate_answer"]:
    """
    Decide what should happen after document grading.

    If web search is needed:
        go to rewrite_question

    Otherwise:
        go directly to generate_answer
    """

    print("\n--- ROUTING AFTER GRADING ---")

    web_search_needed = state["web_search_needed"]

    if web_search_needed:
        print("Decision: Rewrite the question and search the web.")

        return "rewrite_question"

    print("Decision: Generate the final answer directly.")

    return "generate_answer"


# ---------------------------------------------------------
# Build the graph
# ---------------------------------------------------------

workflow = StateGraph(GraphState)


# Add each Python function as a graph node.
workflow.add_node(
    "retrieve_documents",
    retrieve_documents,
)

workflow.add_node(
    "grade_documents",
    grade_documents,
)

workflow.add_node(
    "rewrite_question",
    rewrite_question,
)

workflow.add_node(
    "web_search",
    web_search,
)

workflow.add_node(
    "generate_answer",
    generate_answer,
)


# ---------------------------------------------------------
# Connect the nodes with edges
# ---------------------------------------------------------

# START → retrieve_documents
workflow.add_edge(
    START,
    "retrieve_documents",
)


# retrieve_documents → grade_documents
workflow.add_edge(
    "retrieve_documents",
    "grade_documents",
)


# After grading, choose one of two paths.
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "rewrite_question": "rewrite_question",
        "generate_answer": "generate_answer",
    },
)


# rewrite_question → web_search
workflow.add_edge(
    "rewrite_question",
    "web_search",
)


# web_search → generate_answer
workflow.add_edge(
    "web_search",
    "generate_answer",
)


# generate_answer → END
workflow.add_edge(
    "generate_answer",
    END,
)


# ---------------------------------------------------------
# Compile the graph
# ---------------------------------------------------------

corrective_rag_app = workflow.compile()