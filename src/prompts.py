"""
Prompt templates for the Corrective RAG workflow.
"""

from langchain_core.prompts import ChatPromptTemplate


document_grading_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a document relevance grader.\n\n"
                "Evaluate whether the retrieved document contains "
                "information that can help answer the user's question.\n\n"
                "Return:\n"
                "- binary_score='yes' if the document is relevant\n"
                "- binary_score='no' if the document is irrelevant\n\n"
                "Also provide a short explanation in the reasoning field."
            ),
        ),
        (
            "human",
            (
                "User question:\n"
                "{question}\n\n"
                "Retrieved document:\n"
                "{document}"
            ),
        ),
    ]
)

# Question rewriting prompt template
question_rewriting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You rewrite a user's question to improve document "
                "retrieval and web search. "
                "Preserve the original meaning. "
                "Return only the rewritten question."
            ),
        ),
        (
            "human",
            (
                "Original question:\n"
                "{question}\n\n"
                "Rewrite this question for better search."
            ),
        ),
    ]
)

# ---------------------------------------------------------
# Final answer generation prompt
# ---------------------------------------------------------

answer_generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful question-answering assistant. "
                "Answer the user's question using only the provided context. "
                "If the context does not contain enough information, "
                "say that there is not enough information."
            ),
        ),
        (
            "human",
            (
                "User question:\n"
                "{question}\n\n"
                "Context:\n"
                "{context}\n\n"
                "Provide a clear and concise answer."
            ),
        ),
    ]
)
