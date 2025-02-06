from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from routes import documents, questions
from src.config import get_settings
from schemas import MonitoringData
import time

# load settings
settings = get_settings()

# initialize fastapi app
app = FastAPI(
    title=settings.APP_NAME,
    description="a retrieval-augmented generation (rag) system for answering questions about technical documentation.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "documents",
            "description": "endpoints for uploading and managing documents."
        },
        {
            "name": "questions",
            "description": "endpoints for asking questions and retrieving answers."
        }
    ]
)

# monitoring data
monitoring_data = MonitoringData()

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# middleware to monitor requests
@app.middleware("http")
async def monitor_requests(request, call_next):
    """
    middleware to track request counts and processing times.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    monitoring_data.request_count += 1
    monitoring_data.total_processing_time += process_time

    return response

# include routers
app.include_router(documents.router, prefix="/documents")
app.include_router(questions.router, prefix="/questions")
