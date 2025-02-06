from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    APP_NAME: str = "RAG System"
    MODEL_NAME: str = "llama2"
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Added type annotation
    VECTORSTORE_DIR: str = os.getenv("VECTORSTORE_DIR", "data/vector_store")  # Added type annotation
    API_KEY: str = "6138c5370595af0a0cc290c84a0607ebdc868c90b5f7df354001ed22c86ea52b"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
