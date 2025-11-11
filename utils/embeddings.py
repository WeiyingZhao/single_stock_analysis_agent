"""Embedding utilities for text vectorization."""
from sentence_transformers import SentenceTransformer
from typing import List, Union
from config import Config
import numpy as np


class EmbeddingGenerator:
    """Handles text embedding generation using sentence transformers."""

    def __init__(self, model_name: str = None):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded embedding model: {self.model_name}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            List of floats representing the embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return []

    def compute_similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        try:
            # Convert to numpy arrays if needed
            vec1 = np.array(embedding1) if isinstance(embedding1, list) else embedding1
            vec2 = np.array(embedding2) if isinstance(embedding2, list) else embedding2

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: Union[List[float], np.ndarray],
        candidate_embeddings: List[Union[List[float], np.ndarray]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings from a list of candidates.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top matches to return

        Returns:
            List of tuples (index, similarity_score)
        """
        try:
            similarities = []
            for idx, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((idx, similarity))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top k
            return similarities[:top_k]

        except Exception as e:
            print(f"Error finding most similar: {e}")
            return []


# Global embedding generator instance
_embedding_generator = None


def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get or create the global embedding generator instance.

    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def generate_text_embedding(text: str) -> List[float]:
    """
    Convenience function to generate embedding for text.

    Args:
        text: Input text

    Returns:
        Embedding vector
    """
    generator = get_embedding_generator()
    return generator.generate_embedding(text)


def generate_text_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to generate embeddings for multiple texts.

    Args:
        texts: List of input texts

    Returns:
        List of embedding vectors
    """
    generator = get_embedding_generator()
    return generator.generate_embeddings_batch(texts)


def compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score (0-1)
    """
    generator = get_embedding_generator()
    emb1 = generator.generate_embedding(text1)
    emb2 = generator.generate_embedding(text2)
    return generator.compute_similarity(emb1, emb2)
