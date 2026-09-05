from backend.retriever import search_documents
from backend.gemini import generate_answer


def ask_question(question):

    # Find relevant information from all uploaded documents
    relevant_chunks = search_documents(question, top_k=10)

    # If nothing is found
    if not relevant_chunks:
        return "I could not find the answer in the uploaded documents."

    # Create context with source filename
    context_parts = []

    for chunk in relevant_chunks:

        text = chunk["text"]
        filename = chunk["filename"]

        context_parts.append(
            f"[Source: {filename}]\n{text}"
        )

    # Combine all retrieved chunks
    context = "\n\n".join(context_parts)

    # Ask Gemini
    answer = generate_answer(question, context)

    return answer