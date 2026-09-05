from backend.vector_store import collection
from backend.embeddings import model


def search_documents(question, top_k=20):

    # Convert question into embedding
    question_embedding = model.encode([question])[0]

    # Get more candidates from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved_documents = []

    for i, document in enumerate(documents):

        metadata = {}

        if i < len(metadatas) and metadatas[i] is not None:
            metadata = metadatas[i]

        filename = metadata.get("filename", "Unknown PDF")

        retrieved_documents.append({
            "text": document,
            "filename": filename
        })

    return retrieved_documents