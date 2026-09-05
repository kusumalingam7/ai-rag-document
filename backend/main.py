from backend.rag import ask_question
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from backend.document_loader import read_pdf
from backend.text_splitter import split_text
from backend.embeddings import create_embeddings
from backend.vector_store import store_chunks
app = FastAPI(
    title="AI-Powered RAG Q&A System",
    description="Ask questions about your documents using AI",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Folder where uploaded documents will be stored
DOCUMENTS_FOLDER = Path("documents")
DOCUMENTS_FOLDER.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "OK"
    }
@app.get("/")
def home():
    return {
        "message": "AI-Powered RAG Q&A System is running!"
    }

@app.get("/ask")
def ask(question: str):

    answer = ask_question(question)

    return {
        "question": question,
        "answer": answer
    }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    # Check whether the uploaded file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        return {
            "error": "Only PDF files are supported."
        }

    # Save the PDF
    file_path = DOCUMENTS_FOLDER / file.filename

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    # Read the PDF
    text = read_pdf(file_path)
    chunks = split_text(text)
    embeddings = create_embeddings(chunks)
    stored_count = store_chunks(chunks, embeddings, file.filename)
    return {
    "message": "PDF uploaded successfully!",
    "filename": file.filename,
    "characters_extracted": len(text),
    "chunks_created": len(chunks),
    "chunks_stored": stored_count,
    "text_preview": text[:500]
}