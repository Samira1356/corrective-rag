"""
Test the Bedrock models created in src/llm.py.
"""

from src.llm import get_chat_model, get_embedding_model


print("\n" + "=" * 60)
print("TESTING BEDROCK MODELS")
print("=" * 60)


# Create the embedding model.
embedding_model = get_embedding_model()

print(f"Embedding model type: {type(embedding_model).__name__}")


# Convert a short sentence into a vector.
vector = embedding_model.embed_query(
    "Corrective RAG checks retrieved documents."
)

print(f"Embedding vector length: {len(vector)}")
print(f"First five numbers: {vector[:5]}")


# Create the chat model.
chat_model = get_chat_model()

print(f"\nChat model type: {type(chat_model).__name__}")


# Send a simple message to Bedrock.
response = chat_model.invoke(
    "Explain Corrective RAG in one short sentence."
)

print(f"Bedrock response: {response.content}")
