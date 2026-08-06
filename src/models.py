"""
Structured output models for the Corrective RAG workflow.

These Pydantic classes define the exact response formats
we expect from the Bedrock chat model.
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