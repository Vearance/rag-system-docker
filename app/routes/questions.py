from fastapi import APIRouter, HTTPException
from schemas import QuestionRequest, AnswerResponse
from src.generation import OllamaGenerator
from src.retrieval import VectorStore
import time

router = APIRouter()
generator = OllamaGenerator()
vector_store = VectorStore()

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    handles user questions by retrieving relevant context and generating answers.
    """
    try:
        start_time = time.time()

        # retrieve chunks
        results = vector_store.weighted_query(
            request.question,
            request.history
        )

        # combine context
        context = " ".join([r["chunk"] for r in results])

        # generate response
        answer, generation_time = await generator.response_with_api(
            request.question,
            context
        )

        total_time = time.time() - start_time

        return AnswerResponse(
            answer=answer,
            sources=results,
            metrics={
                "total_time": total_time,
                "retrieval_time": total_time - generation_time,
                "generation_time": generation_time
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
