"""
Evaluate the finalized Corrective RAG configuration on new unseen questions. 

This file checks whether the improvements made during tuning generalize beyond the 
original evaluation questions. 
The correct configuration uses: 
1. k = 5 retrieved chunks.
2. Improved answer-generated prompts. 
3. Bedrock based retrieval evaluation. 
4. Bedrock based final-answer evaluation. 

This file should yse new questions that were not used when choosing k or improving prompt. 

"""
from src.graph import corrective_rag_app
from src.graph_state import GraphState
from src.llm import get_evaluator
from src.prompts import evaluation_prompt

evaluation_data = [
    {
        "question": "How does automation affect administrative costs in healthcare?",
        "expected_answer": (
            "Automation can reduce healthcare administrative costs "
            "by replacing manual transactions with more efficient "
            "electronic workflows."
        ),
    },
    {
        "question": "Why are electronic transactions more efficient than manual ones?",
        "expected_answer": (
            "Electronic transactions reduce manual effort, save time, "
            "and can lower administrative costs."
        ),
    },
    {
        "question": "How can administrative automation benefit healthcare staff?",
        "expected_answer": (
            "Automation can reduce repetitive manual work and free staff "
            "to focus on higher-value activities and patient care."
        ),
    },
    {
        "question": "What financial opportunity exists from improving healthcare administrative automation?",
        "expected_answer": (
            "Greater automation can create substantial cost-saving opportunities "
            "by reducing inefficient manual administrative work."
        ),
    },
    {
        "question": "How can automation improve healthcare workflow efficiency?",
        "expected_answer": (
            "Automation can streamline repetitive administrative processes, "
            "reduce processing time, and improve operational efficiency."
        ),
    },
]

evaluator = get_evaluator()

evaluation_chain = evaluation_prompt | evaluator

scores = []

for item in evaluation_data:
    question = item["question"]

    initial_state : GraphState = {
        "original_question": question, 
        "search_question": question, 
        "documents": [], 
        "filtered_documents":[],
        "web_search_needed":False, 
        "answer":"",
    }

    final_state = corrective_rag_app.invoke(initial_state)

    generated_answer = final_state["answer"]

    evaluation = evaluation_chain.invoke({
        "question":question, 
        "expected_answer": item["expected_answer"],
        "generated_answer": generated_answer
    }
    )

    scores.append(evaluation.score)

    print("\nQuestion:", question)
    print("Score:", evaluation.score)
    print("Reasoning:", evaluation.reasoning)

average_score = sum(scores)/len(scores)

print("\nHoldout average score:", average_score, "/ 3")
