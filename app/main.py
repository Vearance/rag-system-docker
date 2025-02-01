from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, questions
from app.src.config import get_settings
from app.schema import MonitoringData
import time

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/docs",
    redoc_url="/redoc"
)

monitoring_data = MonitoringData()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def monitor_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    monitoring_data.request_count += 1
    monitoring_data.total_processing_time += process_time

    return response

app.include_router(documents.router, prefix="/documents")
app.include_router(questions.router, prefix="/questions")
