"""
Test LLM-based retrieval evaluation for the Corrective RAG project.

This file evaluates whether Chroma retrieves useful PDF chunks
for answering healthcare questions.

The current file performs 6 steps:

1. Creates the Chroma vector store from the CAQH PDF.
2. Creates a retriever that returns the top 3 chunks.
3. Sends each test question to the retriever.
4. Combines the 3 retrieved chunks into one context.
5. Uses Bedrock to evaluate the quality of that retrieved context.
6. Calculates the average retrieval score.

The retrieval score is:

0 = irrelevant
1 = weak
2 = mostly relevant
3 = highly relevant

This test evaluates RETRIEVAL quality.
It does not evaluate the final generated answer.
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

retriever = create_retriever(
    vector_store=vector_store,
    number_of_documents=3,
)

evaluator = get_retrieval_evaluator()

evaluation_chain = retrieval_evaluation_prompt | evaluator

scores = []


for question in questions:

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    evaluation = evaluation_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    scores.append(evaluation.score)

    print("\nQuestion:", question)
    print("Retrieval score:", evaluation.score)
    print("Reasoning:", evaluation.reasoning)


average_score = sum(scores) / len(scores)

print("\nAverage retrieval score:", average_score, "/ 3")