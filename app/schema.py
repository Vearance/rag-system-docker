from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentUpload(BaseModel):
    filename: str = Field(..., example="technical_spec.pdf")
    content_type: str = Field(..., example="application/pdf")

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, example="Explain the engine architecture?")
    history: Optional[List[str]] = Field(
        default=None,
        example=["What is Duke Engine?", "Tell me about cooling systems"]
    )

class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]
    tokens_used: int
    processing_time: float

class MonitoringData(BaseModel):
    request_count: int = 0
    total_processing_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    avg_tokens_used: float = 0.0

    class Config:
        arbitrary_types_allowed = True
