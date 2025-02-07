# import time
from langchain.chains import RetrievalQA
# from langchain.schema import BaseRetriever
from src.config import get_settings
from langchain_community.llms import Ollama

settings = get_settings()
MODEL_NAME = settings.MODEL_NAME

def create_qa_chain(vectorstore, ollama_base_url: str) -> RetrievalQA:
    llm = Ollama(
        base_url=ollama_base_url,
        model=settings.MODEL_NAME
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    return qa_chain


def answer_question(qa_chain: RetrievalQA, question: str) -> str:
    """
    Answers a question using the provided QA chain.
    """
    return qa_chain.run(question)
