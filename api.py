"""
FastAPI interface for the corrective RAG project. 
This file exposes the existing LangGraph Corrective RAG workflow 
through a REST API. 

The current file has 3 main parts: 
1- FastAPI application:
creat the web API. 

2- QuestionRequest: 
Defines the JSON input expected from the user. 

3-ask_question():
Sends the question to the existing Corrective RAG graph
and returns the final answer as JSON.

"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import corrective_rag_app
from src.graph_state import GraphState

# Create the FastAPI application

app = FastAPI(
    title="Corrective RAG API",
    description="Healthcare Corrective RAG API using LangGraph and Amazon Bedrock", 
    version = "1.0.0",
)

# Define the structure of the incoming request
class QuestionRequest(BaseModel):
    question: str

# Creat the /ask API endpoint 
@app.post("/ask")
def ask_question(request:QuestionRequest):

    question = request.question

    initial_state = GraphState = {
        "original_question": question, 
        "search_question": question, 
        "documents": [], 
        "filtered_documents": [], 
        "web_search_needed": False, 
        "answer": "", 

    }

    final_state = corrective_rag_app.invoke(initial_state)

    return {
        "question": question, 
        "answer": final_state["answer"],
    }