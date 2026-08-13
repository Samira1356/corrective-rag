"""
Structured output models for the Corrective RAG workflow.

These Pydantic classes define the exact response formats
we expect from the Bedrock chat model.

The current file has
3 models:
1. GradeDocument: Grades one retrieved document yes/no.
2. EvaluationResult: Scores the findal generated answer. 
3. RetrievalEvaluation: Score the overall retrived context.
"""



from typing import Literal

from pydantic import BaseModel, Field


class GradeDocument(BaseModel):
    """
    Represents the LLM's evaluation of one retrieved document.
    """

    reasoning: str = Field(
        description=(
            "A short explanation of why the document is "
            "relevant or irrelevant to the user's question."
        )
    )

    binary_score: Literal["yes", "no"] = Field(
        description=(
            "Return 'yes' when the document is relevant. "
            "Return 'no' when the document is irrelevant."
        )
    )


class EvaluationResult(BaseModel):
    reasoning: str = Field(
        description="Short explanation of the evaluation"
    )

    score: int = Field(
        description=(
            "Score from 0 to 3. "
            "0=incorrect, 1=partially correct, "
            "2=mostly correct, 3=fully correct"
        )
    )

class RetrievalEvaluation(BaseModel):

    reasoning: str = Field(
        description=(
            "A short explanation of why the retrieval quality"
        )
    )

    score: int = Field(
        description=(
            "Retrieval quality score from 0 to 3. "
            "0= irrelevant, 1=weak, 2=mostly good, 3=highly relevant"
        )
    )



    