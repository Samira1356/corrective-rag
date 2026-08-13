"""
Compare retrieval quality for different k values.

This test check wether retrieving more document chunks improves the quality of
of context returned by chroma 

The current file compares: 
1. k = 3 
2. k = 5

For each setting, Bedrock evaluates the retrieved context and returns a score
from 0 to 3.
"""

from src.vector_store import create_vector_store, create_retriever
from src.llm import get_retrieval_evaluator
from src.prompts import retrieval_evaluation_prompt

questions = [
    "How can automation reduce administrative burden in healthcare?",
    "Why are manual healthcare administrative transactions expensive?",
    "What is one benefit of electronic administrative transactions?",
]

vector_store = create_vector_store()
evaluator = get_retrieval_evaluator()
evaluation_chain = retrieval_evaluation_prompt | evaluator

for k in [3, 5]:
    retriever = create_retriever(
        vector_store=vector_store, 
        number_of_documents= k, 
    )

    scores = []

    print("\nTesting_k=", k)

    for question in questions:
        documents = retriever.invoke(question)

        context = "\n\n".join(
            document.page_content 
            for document in documents
        )

        evaluation = evaluation_chain.invoke(
            {
                "question": question, 
                "context": context
            }
        )

        scores.append(evaluation.score)

        print("question:", question)
        print("Score:",evaluation.score)

    average_score = sum(scores)/len(scores)

    print("Average score for k =", k, ":", average_score)


