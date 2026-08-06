"""
Test the final-answer generation node manually.

LangGraph is not connected yet.
"""

from langchain_core.documents import Document

from src.graph_state import GraphState
from src.nodes import generate_answer


print("\n" + "=" * 60)
print("TESTING FINAL ANSWER GENERATION NODE")
print("=" * 60)


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
    "filtered_documents": [
        Document(
            page_content=(
                "Corrective RAG checks whether retrieved documents "
                "are relevant before generating a final answer. "
                "If the retrieved information is insufficient, "
                "the workflow can perform an external web search."
            ),
            metadata={
                "source": "sample_data",
                "topic": "Corrective RAG",
            },
        )
    ],
    "web_search_needed": True,
    "answer": "",
}


print("\nSTATE BEFORE ANSWER GENERATION")
print("-" * 60)

print(f"Question: {state['original_question']}")

print(
    f"Relevant documents: "
    f"{len(state['filtered_documents'])}"
)

print(f"Answer: {state['answer']}")


answer_update = generate_answer(state)


# Simulate LangGraph merging the returned update.
state.update(answer_update)


print("\nSTATE AFTER ANSWER GENERATION")
print("-" * 60)

print(f"Final answer: {state['answer']}")
