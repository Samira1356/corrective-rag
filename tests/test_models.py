"""
Test the GradeDocument Pydantic model.
"""

from pydantic import ValidationError
from src.models import GradeDocument


print("\n" + "=" * 60)
print("STRUCTURED OUTPUT MODEL")
print("=" * 60)

# Create a valid GradeDocument object.
valid_document = GradeDocument(
    reasoning=(
    "The document explains how Corrective RAG "
        "handles insufficient retrieved information."
    ),

    binary_score="yes",
) 
print("\nValid GradeDocument created successfully.")
print(f"Reasoning: {valid_document.reasoning}")
print(f"Binary score: {valid_document.binary_score}")


print("\n" + "=" * 60) 
print("TESTING INVALID GRADEDOCUMENT")
print("=" * 60) 

try:
    invalid_grade = GradeDocument(
        reasoning="The document may be somewhat relevant .",    
        binary_score="maybe"  # Invalid value, should be "yes" or "no"
    )
except ValidationError as error:
    print("Pydantic correctly rejected the invalid score")
    print(KeyError)
