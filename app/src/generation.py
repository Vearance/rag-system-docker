import httpx
import time
from typing import Tuple
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from src.config import get_settings

settings = get_settings()

class OllamaGenerator:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL

    async def response_with_api(self, query: str, context: str) -> Tuple[str, float]:
        """
        Generates a response using the Ollama API based on the given query and context.

        Args:
            query (str): The user's query.
            context (str): The context retrieved from the vector store.

        Returns:
            Tuple[str, float]: The generated response and the time taken for generation.
        """
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


class OllamaLLM:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def generate(self, prompt: str) -> str:
        """
        Generates a response using the Ollama API.

        Args:
            prompt (str): The prompt to generate a response for.

        Returns:
            str: The generated response.
        """
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

        if response.status_code == 200:
            result = response.json()
            return result['response']
        else:
            raise Exception(f"Error generating response: {response.text}")


def create_qa_chain(vectorstore: BaseRetriever, ollama_base_url: str) -> RetrievalQA:
    """
    Creates a RetrievalQA chain using Ollama as the LLM.

    Args:
        vectorstore (BaseRetriever): The vector store retriever.
        ollama_base_url (str): The base URL for the Ollama API.

    Returns:
        RetrievalQA: The configured QA chain.
    """
    llm = OllamaLLM(ollama_base_url)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore
    )
    return qa_chain


def answer_question(qa_chain: RetrievalQA, question: str) -> str:
    """
    Answers a question using the provided QA chain.

    Args:
        qa_chain (RetrievalQA): The QA chain to use for answering.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
    return qa_chain.run(question)