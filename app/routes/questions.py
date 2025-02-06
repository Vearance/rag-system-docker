from fastapi import APIRouter, HTTPException
from schemas import QuestionRequest, AnswerResponse
from src.generation import create_qa_chain, answer_question
from src.retrieval import load_vectorstore
from src.config import get_settings
import time

settings = get_settings()
VECTORSTORE_DIR = settings.VECTORSTORE_DIR
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL

router = APIRouter()

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Handles user questions by retrieving relevant context and generating answers using Ollama.
    """
    try:
        start_time = time.time()
        
        # Load vector store
        vectorstore = load_vectorstore(VECTORSTORE_DIR)
        if vectorstore is None:
            raise HTTPException(status_code=400, detail="No documents ingested yet.")
        
        # Create QA chain
        qa_chain = create_qa_chain(vectorstore, OLLAMA_BASE_URL)
        
        # Get answer
        answer = answer_question(qa_chain, request.question)
        
        total_time = time.time() - start_time
        
        return AnswerResponse(
            answer=answer,
            sources=[],  # Adjust if sources need to be retrieved
            metrics={
                "total_time": total_time,
                "retrieval_time": 0,  # Adjust if retrieval time is needed
                "generation_time": total_time
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
