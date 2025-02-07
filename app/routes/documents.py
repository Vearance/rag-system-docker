from fastapi import APIRouter, HTTPException, UploadFile, File
from schemas import DocumentUploadResponse
from src.process import DocumentProcessing
from src.retrieval import VectorStore
# import time
from src.config import get_settings
# from langchain_ollama import OllamaEmbeddings
import PyPDF2
from src.embeddings import EmbeddingProcess
import numpy as np

settings = get_settings()
router = APIRouter()
processor = DocumentProcessing()
embedding_process = EmbeddingProcess(model="all-minilm", base_url="http://ollama:11434")
vector_store = VectorStore(embedding_process, processor, use_langchain=True)
VECTORSTORE_DIR = settings.VECTORSTORE_DIR

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith((".txt", ".md", ".pdf")):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        # start_time = time.time()

        # If PDF, use PyPDF2 to extract text
        if file.filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(file.file)
            content = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        else:
            content = await file.read()
            content = content.decode("utf-8")  # Convert bytes to string

        # process document
        processed_data = processor.process_document(content, file.filename)
        chunks = processed_data.get(file.filename, [])

        if not chunks:
            raise HTTPException(status_code=400, detail="No processable content found")

        # create embeddings for each chunk
        try:
            embeddings_list = [embedding_process.encode_query(chunk) for chunk in processed_data[file.filename]]
            embeddings_np = np.array(embeddings_list)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during embedding process: {str(e)}")

        metadata = [
            {
                "filename": file.filename,
                "chunk": chunk_text,
                "document": file.filename,
                "index": i
            }
            for i, chunk_text in enumerate(chunks)
        ]

        # store vector
        vector_store.create_index(embeddings_np, metadata)
        vector_store.save_vectorstore(VECTORSTORE_DIR)

        # processing_time = time.time() - start_time

        return DocumentUploadResponse(
            status="success",
            chunks=len(metadata),
            message="Document processed successfully"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
