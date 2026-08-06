# Corrective RAG with LangGraph, Amazon Bedrock & Chroma

## Overview

This project implements a **Corrective Retrieval-Augmented Generation (Corrective RAG)** workflow using **LangGraph**, **Amazon Bedrock**, **Chroma**, and **LangSmith**.

The workflow retrieves relevant documents, evaluates their relevance with an LLM, decides whether additional retrieval is required, and generates a grounded answer.

This project was built as part of my Agentic AI learning portfolio.

---

## Architecture

```mermaid
flowchart TD
    A[User Question] --> B[retrieve_documents]
    B --> C[Chroma VectorStoreRetriever]
    C --> D[grade_documents]
    D --> E[Amazon Bedrock Nova]
    E --> F[Pydantic Structured Output]
    F --> G{web_search_needed?}

    G -- No --> H[generate_answer]
    G -- Yes --> I[rewrite_question]
    I --> J[Tavily Web Search]
    J --> H

    H --> K[Final Answer]
```

## LangGraph Workflow

```mermaid
flowchart TD
    START --> retrieve_documents
    retrieve_documents --> grade_documents
    grade_documents --> route_after_grading

    route_after_grading -->|Context sufficient| generate_answer
    route_after_grading -->|More information needed| rewrite_question

    rewrite_question --> web_search
    web_search --> generate_answer
    generate_answer --> END
```

## Technologies

- Python
- LangGraph
- LangChain
- Amazon Bedrock (Nova Lite)
- Amazon Titan Embeddings
- Chroma Vector Database
- LangSmith
- Pydantic
- Prompt Engineering

---

## Workflow

```text
User Question
      │
      ▼
Retrieve Documents
      │
      ▼
Grade Documents
      │
      ▼
Web Search Needed?
      │
 ┌────┴────┐
 │         │
No         Yes
 │         │
 ▼         ▼
Generate  Rewrite Question
Answer        │
              ▼
         Web Search
              │
              ▼
        Generate Answer
```

---

## LangSmith Observability

The workflow is traced with LangSmith, including document retrieval, Amazon Bedrock model calls, structured document grading, conditional routing, and final-answer generation.

![LangSmith trace](images/langsmith_trace.png)

## Project Structure

```text
corrective-rag/

├── app.py
├── stream_app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── graph.py
│   ├── graph_state.py
│   ├── llm.py
│   ├── models.py
│   ├── nodes.py
│   ├── prompts.py
│   ├── sample_data.py
│   └── vector_store.py
│
├── tests/
│
└── .gitignore
```

---

## Features

- LangGraph workflow orchestration
- Shared GraphState
- Semantic retrieval using Chroma
- Document grading using Amazon Bedrock
- Question rewriting
- Conditional routing
- Final answer generation
- Streaming execution
- LangSmith tracing
- Component testing

---

## Example Question

```
How does Corrective RAG handle insufficient retrieved information?
```

Example Answer

```
If the retrieved information is insufficient,
Corrective RAG performs an external web search
to retrieve additional relevant information.
```

---

## Future Improvements

- PDF ingestion
- Automatic chunking
- Metadata filtering
- RAG evaluation metrics
- FastAPI deployment
- Docker
- AWS deployment
- Conversation memory

---

## Author

**Samira Khodai**

AI / Machine Learning Engineer

This repository is part of my Agentic AI portfolio built while learning LangGraph, Amazon Bedrock, Retrieval-Augmented Generation (RAG), and production AI workflows.
