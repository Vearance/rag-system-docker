from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DocumentUpload(BaseModel):
    """
    Request model for uploading a document.
    """
    filename: str = Field(..., example="technical_spec.pdf", description="Name of the uploaded file")
    content_type: str = Field(..., example="application/pdf", description="MIME type of the file")

class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload endpoint.
    """
    status: str = Field(example="success", description="Status of the upload process")
    chunks: int = Field(example=10, description="Number of chunks created from the document")
    message: Optional[str] = Field(
        example="Document processed successfully",
        description="Additional message about the upload process"
    )

class QuestionRequest(BaseModel):
    """
    Request model for asking questions.
    """
    question: str = Field(
        ...,
        min_length=3,
        example="Explain the engine architecture?",
        description="The user's question"
    )
    history: Optional[List[str]] = Field(
        default=None,
        example=["What is Duke Engine?", "Tell me about cooling systems"],
        description="List of previous questions for context"
    )

class AnswerResponse(BaseModel):
    """
    Response model for question answering.
    """
    answer: str = Field(
        example="The engine power is 210 HP.",
        description="Generated answer from the RAG system"
    )
    sources: List[Dict[str, str]] = Field(
        example=[{"document": "manual.pdf", "chunk": 5}],
        description="List of sources used to generate the answer"
    )
    metrics: Dict[str, float] = Field(
        example={"total_time": 1.23, "retrieval_time": 0.5, "generation_time": 0.73},
        description="Metrics related to the processing time"
    )

class MonitoringData(BaseModel):
    """
    Model for tracking system performance metrics.
    """
    request_count: int = Field(
        default=0,
        description="Total number of requests processed"
    )
    total_processing_time: float = Field(
        default=0.0,
        description="Total time spent processing requests in seconds"
    )
    success_count: int = Field(
        default=0,
        description="Number of successfully processed requests"
    )
    failure_count: int = Field(
        default=0,
        description="Number of failed requests"
    )
    avg_tokens_used: float = Field(
        default=0.0,
        description="Average number of tokens used per request"
    )

    class Config:
        arbitrary_types_allowed = True