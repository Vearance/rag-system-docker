from typing import Dict, List, Tuple
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

class EmbeddingProcess:
    def __init__(self, model: str = "all-minilm", base_url="http://ollama:11434"):
        self.model_name = model
        self.embedding_model = OllamaEmbeddings(model=model, base_url=base_url)

    def create_embeddings(self, data: Dict[str, List[str]]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Create embeddings for a dictionary of document chunks.

        Args:
            data (Dict[str, List[str]]): Dictionary where keys are filenames and values are lists of text chunks.

        Returns:
            Tuple[np.ndarray, List[Dict]]: A tuple containing:
                - A numpy array of embeddings.
                - A list of metadata dictionaries for each chunk.
        """
        embeddings = []
        metadata = []

        for file_name, chunks in data.items():
            for idx, chunk in enumerate(chunks):
                embedding = self.embedding_model.embed_query(chunk)
                embeddings.append(embedding)
                metadata.append({
                    'document': file_name,
                    'chunk': chunk,
                    'index': idx
                })

        return np.array(embeddings), metadata

    def encode_query(self, text: str) -> np.ndarray:
        """
        Encode a query into an embedding.

        Args:
            text (str): The query text to encode.

        Returns:
            np.ndarray: The normalized embedding vector.
        """
        embedding = self.embedding_model.embed_query(text)
        return embedding / np.linalg.norm(embedding)  # Normalize the embedding

    def create_langchain_documents(self, data: Dict[str, List[str]]) -> List[Document]:
        """
        Convert document chunks into LangChain Document objects.

        Args:
            data (Dict[str, List[str]]): Dictionary where keys are filenames and values are lists of text chunks.

        Returns:
            List[Document]: A list of LangChain Document objects.
        """
        documents = []
        for file_name, chunks in data.items():
            for idx, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        'document': file_name,
                        'chunk_index': idx
                    }
                ))
        return documents
