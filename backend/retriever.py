from backend.vector_store import collection
from backend.embeddings import create_query_embedding


def search_documents(question, top_k=10):

    # Convert the user's question into a Gemini embedding
    question_embedding = create_query_embedding(question)

    # Search ChromaDB for the most relevant document chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved_documents = []

    for i, document in enumerate(documents):

        metadata = {}

        if i < len(metadatas) and metadatas[i] is not None:
            metadata = metadatas[i]

        filename = metadata.get(
            "filename",
            "Unknown PDF"
        )

        retrieved_documents.append({
            "text": document,
            "filename": filename
        })

    return retrieved_documents