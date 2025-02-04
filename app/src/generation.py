import httpx
# import json
# from typing import Dict, List
import time
from src.config import get_settings

settings = get_settings()

class OllamaGenerator:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL

    async def response_with_api(self, query: str, context: str) -> tuple:
        prompt = f"""
        You are an intelligent assistant. Use the context below to answer the query in a concise and accurate manner:

        Context:
        {context}

        Query:
        {query}

        Answer:
        """

        start_time = time.time()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": settings.MODEL_NAME,
                    "prompt": prompt,
                    "max_tokens": settings.MAX_TOKENS,
                    "temperature": settings.TEMPERATURE
                }
            )

        generation_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            return result['response'], generation_time
        else:
            raise Exception(f"Error generating response: {response.text}")
