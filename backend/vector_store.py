import chromadb
import hashlib

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)


def store_chunks(chunks, embeddings, filename="document"):

    ids = []

    for i, chunk in enumerate(chunks):

        unique_text = filename + "_" + str(i) + "_" + chunk

        chunk_id = hashlib.md5(
            unique_text.encode()
        ).hexdigest()
        ids.append(chunk_id)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"filename": filename}
            for _ in chunks
        ]
    )
    return len(chunks)