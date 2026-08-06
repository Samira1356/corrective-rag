"""
Run the Corrective RAG graph with streaming.

This shows:
1. Which node completed
2. What state fields that node updated
3. The final answer
"""

from src.graph import corrective_rag_app
from src.graph_state import GraphState


def create_initial_state(question: str) -> GraphState:
    """
    Create the starting state for one user question.
    """

    cleaned_question = question.strip()

    return {
        "original_question": cleaned_question,
        "search_question": cleaned_question,
        "documents": [],
        "filtered_documents": [],
        "web_search_needed": False,
        "answer": "",
    }


def main() -> None:
    """
    Run the graph and print each node update.
    """

    print("\n" + "=" * 60)
    print("CORRECTIVE RAG STREAMING APPLICATION")
    print("=" * 60)

    question = input("\nEnter your question: ").strip()

    if not question:
        print("\nNo question was entered.")
        return

    initial_state = create_initial_state(question)

    final_answer = ""

    print("\n" + "=" * 60)
    print("GRAPH EXECUTION")
    print("=" * 60)

    for update in corrective_rag_app.stream(
        initial_state,
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():

            print("\n" + "-" * 60)
            print(f"NODE COMPLETED: {node_name}")
            print("-" * 60)

            for field_name, field_value in node_update.items():

                if field_name in {
                    "documents",
                    "filtered_documents",
                }:
                    print(
                        f"{field_name}: "
                        f"{len(field_value)} document(s)"
                    )
                else:
                    print(f"{field_name}: {field_value}")

            if node_name == "generate_answer":
                final_answer = node_update.get("answer", "")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(final_answer)


if __name__ == "__main__":
    main()