"""
Bedrock model configuration for the Corrective RAG workflow.

This file creates reusable Bedrock models.

The current file has:

1. get_chat_model():
   Creates the regular Bedrock Nova chat model.

2. get_embedding_model():
   Creates the Titan embedding model.

3. get_document_grader():
   Creates a structured Bedrock model that returns GradeDocument.

4. get_evaluator():
   Creates a structured Bedrock model that returns EvaluationResult.

5. get_retrieval_evaluator():
   Creates a structured Bedrock model that returns RetrievalEvaluation.
"""

import os

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from src.models import (EvaluationResult, GradeDocument, RetrievalEvaluation)



# Load variables from the project's .env file.
load_dotenv()


# Read the AWS region from .env.
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def get_embedding_model() -> BedrockEmbeddings:
    """
    Create and return the Bedrock embedding model.

    The embedding model converts text into vectors.
    Chroma uses those vectors for similarity search.
    """

    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=AWS_REGION,
    )


def get_chat_model() -> ChatBedrockConverse:
    """
    Create and return the Bedrock chat model.

    The chat model will later grade documents,
    rewrite questions, and generate answers.
    """

    return ChatBedrockConverse(
        model="amazon.nova-lite-v1:0",
        region_name=AWS_REGION,
        temperature=0,
    )

def get_document_grader():
    """
    Create a Bedrock model that returns a GradeDocument object.

    The returned model must follow this structure:

    GradeDocument(
        reasoning="...",
        binary_score="yes" or "no",
    )
    """

    chat_model = get_chat_model()

    structured_grader = chat_model.with_structured_output(
        GradeDocument
    )

    return structured_grader
from src.models import EvaluationResult


def get_evaluator():

    llm = get_chat_model()

    evaluator = llm.with_structured_output(
        EvaluationResult
    )

    return evaluator

def get_retrieval_evaluator():
    """
    Create a structured Bedrock evaluator for retrieval quality.
    """
    llm = get_chat_model()

    evaluator = llm.with_structured_output(
        RetrievalEvaluation
    )

    return evaluator

