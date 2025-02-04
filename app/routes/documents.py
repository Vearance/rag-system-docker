from fastapi import APIRouter, HTTPException
from app.schemas import DocumentUpload, AnswerResponse
from app.src.process import DocumentProcessor
from app.src.embeddings import EmbeddingProcess
from app.src.retrieval import VectorStore
import time

router = APIRouter()
processor = DocumentProcessor()
embedding_process = EmbeddingProcess()
vector_store = VectorStore(embedding_process)

@router.post("/upload", response_model=AnswerResponse)
async def upload_document(doc: DocumentUpload):
    # Auto-validates file type/size
    validated = DocumentUpload(
        filename=doc.filename,
        content_type=doc.content_type
    )

    if "pdf" not in validated.content_type and "markdown" not in validated.content_type:
        raise HTTPException(400, "Invalid file type")

    try:
        start_time = time.time()
        # process document
        processed_data = processor.process_document(doc.content, doc.filename)
        # create embeddings
        embeddings, metadata = embedding_process.create_embeddings(processed_data)
        # store vector
        vector_store.create_index(embeddings, metadata)

        processing_time = time.time() - start_time

        return AnswerResponse(
            filename=doc.filename,
            chunks=len(metadata),
            status="success",
            processing_time=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
