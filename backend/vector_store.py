import chromadb
import hashlib

# Create persistent ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)

# Create or get collection
collection = client.get_or_create_collection(
    name="documents"
)


def store_chunks(chunks, embeddings, filename="document"):

    ids = []

    for i, chunk in enumerate(chunks):

        unique_text = (
            filename
            + "_"
            + str(i)
            + "_"
            + chunk
        )

        chunk_id = hashlib.md5(
            unique_text.encode()
        ).hexdigest()

        ids.append(chunk_id)

    # Store chunks and Gemini embeddings
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"filename": filename}
            for _ in chunks
        ]
    )

    return len(chunks)