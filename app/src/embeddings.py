from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class EmbeddingProcess:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)  # import model

    def create_embeddings(self, data: Dict[str, List[str]]) -> tuple:
        embeddings = []
        metadata = []

        for file_name, chunks in data.items():
            for idx, chunk in enumerate(chunks):
                embedding = self.model.encode(chunk)
                embeddings.append(embedding.tolist())
                metadata.append({
                    'document': file_name,
                    'chunk': chunk,
                    'index': idx
                })

        return np.array(embeddings), metadata

    def encode_query(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)  # encode query
        return embedding / np.linalg.norm(embedding)
