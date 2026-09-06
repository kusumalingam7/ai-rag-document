import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Gemini embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def create_embeddings(chunks):
    embeddings = embeddings_model.embed_documents(chunks)

    return embeddings


def create_query_embedding(question):
    embedding = embeddings_model.embed_query(question)

    return embedding