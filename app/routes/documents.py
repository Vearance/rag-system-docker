from fastapi import APIRouter, HTTPException, UploadFile, File
from schemas import DocumentUploadResponse
from src.process import DocumentProcessing
from src.retrieval import VectorStore
import time
from langchain_ollama import OllamaEmbeddings
import PyPDF2
import os

router = APIRouter()
processor = DocumentProcessing()
embedding_process = OllamaEmbeddings(model="all-minilm", base_url="http://ollama-container:11434")
vector_store = VectorStore(embedding_process, processor)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        start_time = time.time()

        # If PDF, use PyPDF2 to extract text
        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file.file)  # Use file.file instead of bytes
            content = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        else:
            content = await file.read()
            content = content.decode("utf-8")  # Convert bytes to string

        # Process document
        processed_data = processor.process_document(content, file.filename)

        # Create embeddings for each chunk
        try:
            embeddings = [embedding_process.embed_query(chunk) for chunk in processed_data[file.filename]]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during embedding process: {str(e)}")

        metadata = [{"filename": file.filename, "chunk": i} for i in range(len(processed_data[file.filename]))]

        # Store vector
        vector_store.create_index(embeddings, metadata)

        processing_time = time.time() - start_time

        return DocumentUploadResponse(
            status="success",
            chunks=len(metadata),
            message="Document processed successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))