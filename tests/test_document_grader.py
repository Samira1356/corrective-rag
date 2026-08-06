"""
Test the structured Bedrock document grader.

This test checks:

1. A relevant document
2. An irrelevant document
"""

from src.llm import get_document_grader
from src.prompts import document_grading_prompt


print("\n" + "=" * 60)
print("TESTING STRUCTURED DOCUMENT GRADER")
print("=" * 60)


# Create the structured Bedrock grader.
structured_grader = get_document_grader()


# Connect the prompt to the structured grader.
grading_chain = document_grading_prompt | structured_grader


# User question that both documents will be compared against.
question = (
    "How does Corrective RAG handle insufficient "
    "retrieved information?"
)


# This document is relevant to the question.
relevant_document = (
    "Corrective RAG checks retrieved documents for relevance. "
    "If the retrieved information is insufficient, "
    "it can perform an external web search."
)


# This document is unrelated to the question.
irrelevant_document = (
    "A healthy breakfast can include fruit, oatmeal, "
    "eggs, or yogurt."
)


print("\n" + "=" * 60)
print("TEST 1: RELEVANT DOCUMENT")
print("=" * 60)


relevant_result = grading_chain.invoke(
    {
        "question": question,
        "document": relevant_document,
    }
)


print(f"Result type: {type(relevant_result).__name__}")
print(f"Reasoning: {relevant_result.reasoning}")
print(f"Binary score: {relevant_result.binary_score}")


print("\n" + "=" * 60)
print("TEST 2: IRRELEVANT DOCUMENT")
print("=" * 60)


irrelevant_result = grading_chain.invoke(
    {
        "question": question,
        "document": irrelevant_document,
    }
)


print(f"Result type: {type(irrelevant_result).__name__}")
print(f"Reasoning: {irrelevant_result.reasoning}")
print(f"Binary score: {irrelevant_result.binary_score}")