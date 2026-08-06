"""
Node function for the Corrective RAG workflow.

A LnagGraph node is a python function that: 

1- Receive the current graph state. 
2- Perform one specific task. 
3- Return a dictionary containing state updates.

"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch


from src.graph_state import GraphState
from src.llm import get_chat_model, get_document_grader
#from src.prompts import document_grading_prompt
from src.prompts import (
    answer_generation_prompt,
    document_grading_prompt,
    question_rewriting_prompt,
)
from src.vector_store import create_retriever, create_vector_store

load_dotenv()


# Create a reusable objects

# Build the Chroma vector store using the sample documents. 
vector_store = create_vector_store()

# Create a retriever connect to vector store.
retriever = create_retriever(
    vector_store=vector_store,
    number_of_documents=3,
    #search_type="similarity"
    )
# Create the structured bedrock grader. 
structured_grader = get_document_grader()

# Connect the grading prompt to the structured grader.
grading_chain = document_grading_prompt | structured_grader

# Create a regular Bedrock chat model for rewriting questions.
chat_model = get_chat_model()


# Connect the rewriting prompt to the chat model.
question_rewriting_chain = (
    question_rewriting_prompt
    | chat_model
)

# Connect the answer-generation prompt to Bedrock Nova.
answer_generation_chain = (
    answer_generation_prompt
    | chat_model
)

# Create a Tavily web search tool.
web_search_tool = TavilySearch(
    max_results=3,
    topic="general",
    include_answer=False,
    include_raw_content=False,
)

# Node 1: Retrieve documents 
def retrieve_documents(state: GraphState) -> dict:
    """
    Retrieve documents related to the current search question.

    Reads from the state:
        search_question 
    Returns an updated for: 
        documents
    """
    print("\n---RETRIEVING DOCUMENTS NODE---")
    question = state["search_question"]

    print(f"search_question: {question}")

    retrieved_documents = retriever.invoke(question)
    print(f"number of retrieved documents: {len(retrieved_documents)}")

    return {"documents": retrieved_documents}


# Node 2: Grade documents
#If at least one relevant document remains → generate the answer
# If zero relevant documents remain → rewrite the question and search the web
# ---------------------------------------------------------
# Node 2: Grade documents
# ---------------------------------------------------------

def grade_documents(state: GraphState) -> dict:
    """
    Grade each retrieved document for relevance.

    Reads from the state:
        original_question
        documents

    Returns updates for:
        filtered_documents
        web_search_needed
    """

    print("\n--- GRADING DOCUMENTS NODE ---")

    # Read the question and retrieved documents
    # from the shared graph state.
    question = state["original_question"]
    documents = state["documents"]

    # Start with an empty list.
    # Relevant documents will be added here.
    filtered_documents: list[Document] = []

    # Grade every retrieved document.
    for number, document in enumerate(
        documents,
        start=1,
    ):
        print(f"\nGrading document {number}...")

        grade = grading_chain.invoke(
            {
                "question": question,
                "document": document.page_content,
            }
        )

        print(f"Reasoning: {grade.reasoning}")
        print(f"Binary score: {grade.binary_score}")

        if grade.binary_score.lower() == "yes":
            filtered_documents.append(document)

    # Search the web only when no relevant documents remain.
    web_search_needed = len(filtered_documents) == 0

    print(
        f"\nRelevant documents kept: "
        f"{len(filtered_documents)}"
    )

    print(
        f"Web search needed: "
        f"{web_search_needed}"
    )

    return {
        "filtered_documents": filtered_documents,
        "web_search_needed": web_search_needed,
    }
# ---------------------------------------------------------
# Node 3: Rewrite question
# ---------------------------------------------------------

def rewrite_question(state: GraphState) -> dict:
    """
    Rewrite the original question for a better search.

    Reads from the state:
        original_question

    Returns an update for:
        search_question
    """

    print("\n--- REWRITING QUESTION NODE ---")

    original_question = state["original_question"]

    print(f"Original question: {original_question}")

    response = question_rewriting_chain.invoke(
        {
            "question": original_question,
        }
    )

    rewritten_question = response.content.strip()

    print(f"Rewritten question: {rewritten_question}")

    return {
        "search_question": rewritten_question,
    }

    # ---------------------------------------------------------
# Node 4: Web search
# ---------------------------------------------------------

def web_search(state: GraphState) -> dict:
    """
    Search the web using the rewritten search question.

    Reads from the state:
        search_question
        filtered_documents

    Returns an update for:
        filtered_documents
    """

    print("\n--- WEB SEARCH NODE ---")

    search_question = state["search_question"]

    print(f"Search question: {search_question}")

    # Copy the existing relevant documents.
    updated_documents = state["filtered_documents"].copy()

    # Send one dictionary input to the Tavily tool.
    search_response = web_search_tool.invoke(
        {
            "query": search_question,
        }
    )

    # Tavily returns a dictionary containing a results list.
    search_results = search_response.get("results", [])

    print(
        f"Number of web results: "
        f"{len(search_results)}"
    )

    for number, result in enumerate(
        search_results,
        start=1,
    ):
        title = result.get(
            "title",
            "Untitled web result",
        )

        url = result.get(
            "url",
            "",
        )

        content = result.get(
            "content",
            "",
        )

        print(f"\nWeb result {number}")
        print(f"Title: {title}")
        print(f"URL: {url}")

        # Convert each Tavily result into a LangChain Document.
        web_document = Document(
            page_content=content,
            metadata={
                "source": url,
                "title": title,
                "type": "web_search",
            },
        )

        updated_documents.append(web_document)

    print(
        f"\nTotal documents after web search: "
        f"{len(updated_documents)}"
    )

    return {
        "filtered_documents": updated_documents,
    }
# ---------------------------------------------------------
# Node 5: Generate final answer
# ---------------------------------------------------------

def generate_answer(state: GraphState) -> dict:
    """
    Generate the final answer using the relevant documents.

    Reads from the state:
        original_question
        filtered_documents

    Returns an update for:
        answer
    """

    print("\n--- GENERATING FINAL ANSWER NODE ---")

    question = state["original_question"]
    documents = state["filtered_documents"]

    print(f"Original question: {question}")
    print(f"Number of context documents: {len(documents)}")

    # Combine the text from all relevant documents.
    context_parts = []

    for number, document in enumerate(
        documents,
        start=1,
    ):
        context_parts.append(
            f"Document {number}:\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    response = answer_generation_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    final_answer = response.content.strip()

    print(f"Final answer: {final_answer}")

    return {
        "answer": final_answer,
    }
