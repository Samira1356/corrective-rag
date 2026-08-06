"""
Test the web-search node manually.

LangGraph is not connected yet.
"""

import os

from dotenv import load_dotenv

from src.graph_state import GraphState
from src.nodes import web_search


load_dotenv()


print("\n" + "=" * 60)
print("TESTING WEB SEARCH NODE")
print("=" * 60)


# ---------------------------------------------------------
# Check the Tavily API key
# ---------------------------------------------------------

tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()

placeholder_values = {
    "",
    "replace_with_your_actual_tavily_key",
    "your_tavily_api_key",
    "tvly-your-real-key",
}

if tavily_api_key.lower() in placeholder_values:
    raise ValueError(
        "A valid Tavily API key is not configured. "
        "Create a Tavily account and replace the placeholder in .env."
    )

print("Tavily API key is configured.")
# ---------------------------------------------------------
# Create a sample graph state
# ---------------------------------------------------------

state: GraphState = {
    "original_question": (
        "How does Corrective RAG handle "
        "insufficient retrieved information?"
    ),
    "search_question": (
        "What strategies does Corrective RAG employ "
        "to address insufficient retrieved information?"
    ),
    "documents": [],
    "filtered_documents": [],
    "web_search_needed": True,
    "answer": "",
}


print("\nSTATE BEFORE WEB SEARCH")
print("-" * 60)

print(f"Search question: {state['search_question']}")

print(
    f"Filtered documents: "
    f"{len(state['filtered_documents'])}"
)


# ---------------------------------------------------------
# Run the web-search node
# ---------------------------------------------------------

web_search_update = web_search(state)


# Simulate LangGraph updating the state.
state.update(web_search_update)


print("\nSTATE AFTER WEB SEARCH")
print("-" * 60)

print(
    f"Total filtered documents: "
    f"{len(state['filtered_documents'])}"
)


for number, document in enumerate(
    state["filtered_documents"],
    start=1,
):
    print(f"\nDocument {number}")
    print(f"Type: {document.metadata.get('type')}")
    print(f"Title: {document.metadata.get('title')}")
    print(f"Source: {document.metadata.get('source')}")
    print(f"Content: {document.page_content}")