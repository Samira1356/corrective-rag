from src.graph import corrective_rag_app
from src.graph_state import GraphState
from src.llm import get_evaluator
from src.prompts import evaluation_prompt


evaluation_data = [
    {
        "question": "How can automation reduce administrative burden in healthcare?",
        "expected_answer": (
            "Automation can reduce manual administrative work by increasing "
            "electronic processing and reducing costly manual transactions."
        ),
    },
    {
        "question": "Why are manual healthcare administrative transactions expensive?",
        "expected_answer": (
            "Manual transactions require more staff time and operational "
            "effort than electronic transactions."
        ),
    },
    {
        "question": "What is one benefit of electronic administrative transactions?",
        "expected_answer": (
            "Electronic transactions can reduce administrative costs "
            "and improve efficiency."
        ),
    },
    {
        "question": "What role does automation play in healthcare administration?",
        "expected_answer": (
            "Automation helps replace repetitive manual administrative "
            "processes with more efficient electronic workflows."
        ),
    },
    {
        "question": "What type of healthcare work can benefit from greater automation?",
        "expected_answer": (
            "Administrative processes such as eligibility, claims, and "
            "other transactions can benefit from greater automation."
        ),
    },
]


evaluator = get_evaluator()

evaluation_chain = evaluation_prompt | evaluator

scores = []


for item in evaluation_data:

    question = item["question"]

    initial_state: GraphState = {
        "original_question": question,
        "search_question": question,
        "documents": [],
        "filtered_documents": [],
        "web_search_needed": False,
        "answer": "",
    }

    final_state = corrective_rag_app.invoke(initial_state)

    generated_answer = final_state["answer"]

    evaluation = evaluation_chain.invoke(
        {
            "question": question,
            "expected_answer": item["expected_answer"],
            "generated_answer": generated_answer,
        }
    )

    scores.append(evaluation.score)

    print("\nQuestion:", question)
    print("Expected answer:", item["expected_answer"])
    print("Generated answer:", generated_answer)
    print("Score:", evaluation.score)
    print("Reasoning:", evaluation.reasoning)

average_score = sum(scores) / len(scores)

print("\nAverage evaluation score:", average_score, "/ 3")