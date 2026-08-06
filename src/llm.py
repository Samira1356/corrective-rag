"""
Bedrock model configuration for the Corrective RAG project.

This file creates:

1. An embedding model
   Converts text into numerical vectors.

2. A chat model
   Reads prompts and generates responses.
"""

import os

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from src.models import GradeDocument


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

