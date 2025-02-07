from fastapi import APIRouter, HTTPException
from schemas import QuestionRequest, AnswerResponse
from src.generation import create_qa_chain, answer_question
from src.retrieval import load_vectorstore
from src.config import get_settings
import time
from src.embeddings import EmbeddingProcess

settings = get_settings()
VECTORSTORE_DIR = settings.VECTORSTORE_DIR
OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL

router = APIRouter()

embedding_process = EmbeddingProcess(model="all-minilm")

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Handles user questions by retrieving relevant context and generating answers using Ollama.
    """
    try:
        start_time = time.time()

        # load vector store
        vectorstore = load_vectorstore(
            VECTORSTORE_DIR,
            use_langchain=True,
            embedding_process=embedding_process
        )
        if vectorstore is None:
            raise HTTPException(status_code=400, detail="No documents ingested yet.")

        # create QA chain
        qa_chain = create_qa_chain(vectorstore, OLLAMA_BASE_URL)

        # get answer
        answer = answer_question(qa_chain, request.question)

        total_time = time.time() - start_time

        return AnswerResponse(
            answer=answer,
            sources=[],
            metrics={
                "total_time": total_time,
                "retrieval_time": 0,
                "generation_time": total_time
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
