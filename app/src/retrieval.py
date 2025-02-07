# import faiss
import numpy as np
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.embeddings import EmbeddingProcess
from src.process import DocumentProcessing

class VectorStore:
    """
    A class for managing the vector store and performing similarity searches.
    Supports both custom FAISS and LangChain FAISS backends.
    """

    def __init__(
        self,
        embedding_process: EmbeddingProcess,
        document_process: DocumentProcessing,
        use_langchain: bool = False
    ):
        """
        Initializes the vector store with embedding and document processing utilities.

        Args:
            embedding_process (EmbeddingProcess): The embedding process utility.
            document_process (DocumentProcessing): The document processing utility.
            use_langchain (bool): Whether to use LangChain's FAISS backend. Defaults to False.
        """
        self.embed_process = embedding_process
        self.doc_process = document_process
        self.use_langchain = use_langchain
        self.index = None
        self.metadata = []
        self.vectorstore = None

    def create_index(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Creates a FAISS index from the provided embeddings and metadata.

        Args:
            embeddings (np.ndarray): The embeddings to index.
            metadata (List[Dict]): Metadata associated with each embedding.
        """
        self.metadata = metadata

        if self.use_langchain:
            # Convert embeddings and metadata into LangChain Documents
            documents = []
            for i, embed in enumerate(embeddings):
                doc = Document(
                    page_content=metadata[i]["chunk"],
                    metadata={
                        "document": metadata[i]["document"],
                        "index": metadata[i]["index"]
                    }
                )
                documents.append(doc)

            # Create LangChain FAISS vector store
            self.vectorstore = FAISS.from_documents(
                documents=documents,  # Explicitly name the 'documents' argument
                embedding=self.embed_process.embedding_model,  # Explicitly name the 'embedding' argument
                # metadatas=[doc.metadata for doc in documents]
            )
        # else:
            # # Normalize embeddings to unit vectors
            # norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # normalized_embeddings = embeddings / norms

            # # Create custom FAISS index
            # dim = embeddings.shape[1]
            # self.index = FAISS.IndexFlatIP(dim)  # Cosine similarity
            # self.index.add(normalized_embeddings)

    def weighted_query(
        self,
        query: str,
        history: Optional[List[str]] = None,
        top_k: int = 3,
        current_weight: float = 0.7
    ) -> List[Dict]:
        """
        Performs a weighted search using the query and optional history.
        Returns the top-k most relevant chunks.

        """
        query_cleaned = self.doc_process.preprocess_text(query)
        query_embed = self.embed_process.encode_query(query_cleaned)  # Encode query

        if history:
            history_embed = self.embed_process.encode_query(" ".join(history))
        else:
            history_embed = np.zeros_like(query_embed)

        # Combine query and history with weights
        combined_query = current_weight * query_embed + (1 - current_weight) * history_embed

        if self.use_langchain:
            # Use LangChain's FAISS for retrieval
            results = self.vectorstore.similarity_search_by_vector(
                combined_query, k=top_k
            )
            return [
                {
                    "chunk": doc.page_content,
                    "document": doc.metadata["document"],
                    "index": doc.metadata["index"],
                    "distance": 1.0  # LangChain doesn't return distances by default
                }
                for doc in results
            ]
        else:
            # Use custom FAISS for retrieval
            svec = np.array(combined_query).reshape(1, -1)
            distances, pos = self.index.search(svec, k=top_k)

            # Retrieve data
            results = []
            for i, idx in enumerate(pos[0]):
                results.append({
                    "chunk": self.metadata[idx]["chunk"],
                    "document": self.metadata[idx]["document"],
                    "index": self.metadata[idx]["index"],
                    "distance": float(distances[0][i])
                })
            return results

    def save_vectorstore(self, directory: str):
        """
        Saves the vector store to disk.

        Args:
            directory (str): The directory to save the vector store to.
        """
        if self.use_langchain and self.vectorstore:
            self.vectorstore.save_local(directory)
        else:
            raise NotImplementedError("Saving is only supported for LangChain FAISS.")


# Standalone function to load the vectorstore (instead of a method inside the class)
def load_vectorstore(directory: str, use_langchain: bool = False, embedding_process=None, document_process=None):
    """
    Loads the vector store from disk.
    """
    vectorstore = None

    if use_langchain:
        vectorstore = FAISS.load_local(
            directory,
            embedding_process.embedding_model,
            allow_dangerous_deserialization=True  # <---- ADD THIS LINE: allow_dangerous_deserialization=True
        )
    else:
        raise NotImplementedError("Loading is only supported for LangChain FAISS.")

    return vectorstore
