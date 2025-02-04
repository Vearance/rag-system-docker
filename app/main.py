from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from app.routes import documents, questions
from app.src.config import get_settings
from app.schemas import MonitoringData
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# api key authentication
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    """
    validates the api key provided in the request header.
    raises an http 403 error if the key is invalid.
    """
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="invalid api key")
    return api_key

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
app.include_router(documents.router, prefix="/documents", dependencies=[Depends(get_api_key)])
app.include_router(questions.router, prefix="/questions", dependencies=[Depends(get_api_key)])