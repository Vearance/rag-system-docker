from pydantic import BaseSettings
from functools import lru_cache
# from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "RAG System"
    MODEL_NAME: str = "llama2"  # generation model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # embedding model
    VECTOR_STORE_PATH: str = "./data/vector_store"
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    OLLAMA_BASE_URL: str = "http://ollama:11434"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
