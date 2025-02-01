import faiss
import numpy as np
from typing import List, Dict
from app.src.embeddings import EmbeddingProcess
from app.src.process import DocumentProcessing

class VectorStore:
    def __init__(self, embedding_process: EmbeddingProcess, document_process: DocumentProcessing):
        self.embed_process = embedding_process
        self.doc_process = document_process
        self.index = None
        self.metadata = []

    def create_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        self.metadata = metadata

        # normalize embeddings to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # divide each embedding by its norm to get unit vectors
        normalized_embeddings = embeddings / norms

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(normalized_embeddings)

    def weighted_query(self, query: str, history: List[str] = None, top_k: int = 3, current_weight: float = 0.7) -> List[Dict]:
        query_cleaned = self.doc_process.preprocess_text(query)
        query_embed = self.embed_process.encode_query(query_cleaned)  # encode query

        if history:
            history_embed = self.embed_process.encode_query(" ".join(history))
        else:
            history_embed = np.zeros_like(query_embed)

        # weight
        combined_query = current_weight * query_embed + (1 - current_weight) * history_embed

        svec = np.array(combined_query).reshape(1, -1)
        distances, pos = self.index.search(svec, k=top_k)

        # retrieve data
        results = []
        for i, idx in enumerate(pos[0]):
            results.append({
                "chunk": self.metadata[idx]["chunk"],
                "document": self.metadata[idx]["document"],
                "index": self.metadata[idx]["index"],
                "distance": float(distances[0][i])
            })
        return results
