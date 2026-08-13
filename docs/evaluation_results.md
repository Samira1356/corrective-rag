# Corrective RAG Evaluation Results

## Retrieval Evaluation

Initial retrieval configuration:

- k = 3
- Average retrieval score: 2.33 / 3

Improved retrieval configuration:

- k = 5
- Average retrieval score: 3.0 / 3

Conclusion:

Increasing the number of retrieved chunks from 3 to 5 improved retrieval quality for the evaluation questions.

## Answer Evaluation

Initial answer-generation prompt:

- Average answer score: 2.4 / 3

Improved answer-generation prompt:

- Average answer score: 3.0 / 3

Conclusion:

The initial prompt produced answers that were often correct but too brief.

The updated prompt asked the model to include important details and examples from the retrieved context.

This improved answer completeness.

## Holdout Evaluation

The finalized configuration was tested using new questions that were not used during tuning.

Final configuration:

- Retriever k = 5
- Improved answer-generation prompt
- Bedrock Nova for generation and evaluation

Holdout average score:

- 3.0 / 3

## Important Limitation

A perfect evaluation score does not mean the system is universally 100% accurate.

The evaluation dataset is still small, and an LLM is being used as the evaluator.

Future evaluation should include:

- More questions
- Difficult factual questions
- Unsupported questions
- Multi-part questions
- Retrieval precision and recall
- Faithfulness evaluation
