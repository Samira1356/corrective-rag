from src.vector_store import create_vector_store, create_retriever


evaluation_data = [
    {
        "question": "How can automation reduce administrative burden in healthcare?",
        "expected_keyword": "automation",
    },
    {
        "question": "Why are manual healthcare administrative transactions expensive?",
        "expected_keyword": "manual",
    },
    {
        "question": "What is one benefit of electronic administrative transactions?",
        "expected_keyword": "electronic",
    },
]


vector_store = create_vector_store()

retriever = create_retriever(
    vector_store=vector_store,
    number_of_documents=3,
)


scores = []


for item in evaluation_data:

    retrieved_documents = retriever.invoke(
        item["question"]
    )

    retrieved_text = " ".join(
        document.page_content.lower()
        for document in retrieved_documents
    )

    keyword_found = (
        item["expected_keyword"].lower()
        in retrieved_text
    )

    score = 1 if keyword_found else 0

    scores.append(score)

    print("Question:", item["question"])
    print("Expected keyword:", item["expected_keyword"])
    print("Retrieved correctly:", keyword_found)


average_score = sum(scores) / len(scores)

print(
    "\nAverage retrieval score:",
    average_score,
    "/ 1",
)