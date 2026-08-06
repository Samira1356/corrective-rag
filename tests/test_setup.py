"""
Test the Corrective RAG project setup.

This file checks:

1. Whether the required Python libraries can be imported.
2. Whether the .env file can be loaded.
3. Whether the Tavily API key exists.
4. Whether Amazon Bedrock can answer a simple question.
"""

# ============================================================
# SECTION 1: STANDARD PYTHON IMPORTS
# ============================================================

import os


# ============================================================
# SECTION 2: THIRD-PARTY IMPORTS
# ============================================================

# load_dotenv() reads variables from the local .env file.
from dotenv import load_dotenv

# BaseModel and Field will later define structured LLM output.
from pydantic import BaseModel, Field

# Document represents one piece of retrieved information.
from langchain_core.documents import Document

# AWS Bedrock chat model and embedding model.
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

# Chroma is our local vector database.
from langchain_chroma import Chroma

# TavilySearch performs external web searches.
from langchain_tavily import TavilySearch

# LangGraph classes used to create the workflow.
from langgraph.graph import StateGraph, START, END


# ============================================================
# SECTION 3: LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


# ============================================================
# SECTION 4: CHECK ENVIRONMENT VARIABLES
# ============================================================
# ============================================================
# SECTION 4: CHECK TAVILY CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("CHECK 1: TAVILY CONFIGURATION")
print("=" * 60)

if tavily_api_key is None:
    print("❌ TAVILY_API_KEY is missing from the .env file.")

elif tavily_api_key == "replace_with_your_actual_tavily_key":
    print("⚠️ Placeholder API key detected.")
    print("Please replace it with your real Tavily API key before using web search.")

else:
    print("✅ Tavily API key appears to be configured.")

print(f"AWS region: {aws_region}")


# ============================================================
# SECTION 5: CHECK IMPORTED CLASSES
# ============================================================

print("\n" + "=" * 60)
print("CHECK 2: REQUIRED LIBRARIES")
print("=" * 60)

print(f"Document: {Document}")
print(f"ChatBedrockConverse: {ChatBedrockConverse}")
print(f"BedrockEmbeddings: {BedrockEmbeddings}")
print(f"Chroma: {Chroma}")
print(f"TavilySearch: {TavilySearch}")
print(f"StateGraph: {StateGraph}")
print(f"START constant: {START}")
print(f"END constant: {END}")

print("\nAll required imports succeeded.")


# ============================================================
# SECTION 6: CREATE THE BEDROCK LLM
# ============================================================

llm = ChatBedrockConverse(
    model="amazon.nova-lite-v1:0",
    region_name=aws_region,
    temperature=0,
)


# ============================================================
# SECTION 7: TEST THE BEDROCK LLM
# ============================================================

print("\n" + "=" * 60)
print("CHECK 3: AMAZON BEDROCK")
print("=" * 60)

try:
    response = llm.invoke(
        "Explain retrieval-augmented generation in one short sentence."
    )

    print("Bedrock connection succeeded.")
    print(f"Model response: {response.content}")

except Exception as error:
    print("Bedrock connection failed.")
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {error}")


# ============================================================
# SECTION 8: FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SETUP TEST FINISHED")
print("=" * 60)
