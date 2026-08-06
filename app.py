"""
Main application file for the corrective RAG system.

The application: 
1- Accepts a question from the user.
2- Create an initial graph state based on the question.
3- Run the cmpiled LangGraph workflow using the initial graph state.
4- Display the final answer to the user.

"""
from src.graph import corrective_rag_app
from src.graph_state import GraphState

def create_initial_state(question: str) -> GraphState:
    """
    Create the starting state for one user question.

    Parameters
    ----------
    question : str
        The question entered by the user.

    Returns
    -------
    GraphState
        A complete initial state dictionary.
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

def run_corrective_rag(question: str) -> GraphState:
    """
    Run the complete Corrective RAG workflow.

    Parameters
    ----------
    question: 
        The user's question.

    Returns
    -------
    GraphState:
        The final state returned by LangGraph. 
    """
    initial_state = create_initial_state(question)
    final_state = corrective_rag_app.invoke(initial_state)
    return final_state

def main() -> None:
    """
    Start the terminal application. 
    """
    print("n" + "=" * 60)
    print("Corrective RAG application")
    print("=" * 60)

    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered. Exiting.")
        return
    final_state = run_corrective_rag(question)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(f"Original question: {final_state['original_question']}")

    print(f"Retrieved documents: {len(final_state['documents'])}")

    print(f"Relevant context documents: {len(final_state['filtered_documents'])}")

    print(f"Web search needed: {final_state['web_search_needed']}")

    print(f"Final answer: {final_state['answer']}")

if __name__ == "__main__":
    main()
       