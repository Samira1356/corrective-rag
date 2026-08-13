"""
Prompt templates for the Corrective RAG workflow.

These prompts define the instructions sent to the Bedrock chat model.
The current file has 5 prompts:
1. document_grading_prompt: Grades one retrieved document yes/no relevant or irrelevant.
2. question_rewriting_prompt: Rewrites the user's question for better retrieval or web search.
3. answer_generation_prompt: Generates the final answer using the retrieved context.
4. evaluation_prompt: Evaluates the correctness of the final generated answer.
5. retrieval_evaluation_prompt: Evaluates the quality of the retrieved context. 
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
                "Provide a clear and complete answer. "
                "Include the important details and examples from the context "
                "that directly help answer the question. "
                "Do not omit relevant information just to make the answer shorter."
)
        ),
    ]
)

evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an evaluator. "
                "Compare the generated answer with the expected answer. "
                "Score correctness from 0 to 3."
            ),
        ),
        (
            "human",
            (
                "Question:\n{question}\n\n"
                "Expected answer:\n{expected_answer}\n\n"
                "Generated answer:\n{generated_answer}"
            ),
        ),
    ]
)

retrieval_evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are evaluating retrieval quality. "
                "Determine whether the retrieved context contains "
                "useful information for answering the user's question. "
                "Score the retrieval from 0 to 3."
            ),
        ),
        (
            "human",
            (
                "Question:\n"
                "{question}\n\n"
                "Retrieved context:\n"
                "{context}"
            ),
        ),
    ]
)