"""Vector store wrapper for Chroma database."""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from langchain_core.tools import tool
import json
from config import Config


class VectorStoreManager:
    """Manager class for Chroma vector database operations."""

    def __init__(self, persist_directory: str = None, collection_name: str = None):
        """
        Initialize the vector store manager.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or Config.COLLECTION_NAME

        # Initialize Chroma client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Tesla stock event profiles"}
        )

    def add_event_profile(
        self,
        event_id: str,
        embedding: List[float],
        metadata: Dict,
        document: str
    ) -> bool:
        """
        Add an event profile to the vector store.

        Args:
            event_id: Unique identifier for the event
            embedding: Vector embedding of the event
            metadata: Metadata dictionary (date, price_change, etc.)
            document: Text description of the event

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.add(
                ids=[event_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document]
            )
            return True
        except Exception as e:
            print(f"Error adding event profile: {e}")
            return False

    def add_event_profiles_batch(
        self,
        event_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        documents: List[str]
    ) -> bool:
        """
        Add multiple event profiles in batch.

        Args:
            event_ids: List of event IDs
            embeddings: List of embeddings
            metadatas: List of metadata dictionaries
            documents: List of document strings

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.add(
                ids=event_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            return True
        except Exception as e:
            print(f"Error adding batch event profiles: {e}")
            return False

    def query_similar_events(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        similarity_threshold: float = None
    ) -> Dict:
        """
        Query for similar events based on embedding.

        Args:
            query_embedding: Vector embedding to search for
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score (optional)

        Returns:
            Dictionary with query results
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Filter by similarity threshold if provided
            threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD

            filtered_results = {
                'ids': [],
                'distances': [],
                'metadatas': [],
                'documents': []
            }

            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    # Convert distance to similarity (cosine similarity)
                    similarity = 1 - distance

                    if similarity >= threshold:
                        filtered_results['ids'].append(results['ids'][0][i])
                        filtered_results['distances'].append(distance)
                        filtered_results['metadatas'].append(results['metadatas'][0][i])
                        filtered_results['documents'].append(results['documents'][0][i])

            return {
                'success': True,
                'n_results': len(filtered_results['ids']),
                'results': filtered_results
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'n_results': 0,
                'results': None
            }

    def get_event_by_id(self, event_id: str) -> Optional[Dict]:
        """
        Retrieve a specific event by ID.

        Args:
            event_id: Event identifier

        Returns:
            Event data dictionary or None
        """
        try:
            results = self.collection.get(
                ids=[event_id],
                include=['embeddings', 'metadatas', 'documents']
            )

            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'embedding': results['embeddings'][0] if results['embeddings'] else None,
                    'metadata': results['metadatas'][0] if results['metadatas'] else None,
                    'document': results['documents'][0] if results['documents'] else None
                }
            return None

        except Exception as e:
            print(f"Error getting event by ID: {e}")
            return None

    def count_events(self) -> int:
        """
        Count total number of events in the collection.

        Returns:
            Number of events
        """
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error counting events: {e}")
            return 0

    def delete_event(self, event_id: str) -> bool:
        """
        Delete an event from the collection.

        Args:
            event_id: Event identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[event_id])
            return True
        except Exception as e:
            print(f"Error deleting event: {e}")
            return False

    def clear_collection(self) -> bool:
        """
        Clear all events from the collection.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Tesla stock event profiles"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False


# Tool wrappers for LangChain agents

@tool
def store_event_profile(event_data: str) -> str:
    """
    Store an event profile in the vector database.

    Args:
        event_data: JSON string containing event_id, embedding, metadata, and document

    Returns:
        Success message or error
    """
    try:
        data = json.loads(event_data)
        vector_store = VectorStoreManager()

        success = vector_store.add_event_profile(
            event_id=data['event_id'],
            embedding=data['embedding'],
            metadata=data['metadata'],
            document=data['document']
        )

        if success:
            return f"Successfully stored event profile: {data['event_id']}"
        else:
            return "Failed to store event profile"

    except Exception as e:
        return f"Error storing event profile: {str(e)}"


@tool
def search_similar_events(query_data: str) -> str:
    """
    Search for similar historical events based on embedding.

    Args:
        query_data: JSON string containing query_embedding and optional n_results

    Returns:
        JSON string with similar events
    """
    try:
        data = json.loads(query_data)
        vector_store = VectorStoreManager()

        results = vector_store.query_similar_events(
            query_embedding=data['query_embedding'],
            n_results=data.get('n_results', 5),
            similarity_threshold=data.get('similarity_threshold')
        )

        return json.dumps(results, indent=2)

    except Exception as e:
        return f"Error searching similar events: {str(e)}"


@tool
def get_vector_store_stats() -> str:
    """
    Get statistics about the vector store.

    Returns:
        JSON string with store statistics
    """
    try:
        vector_store = VectorStoreManager()
        count = vector_store.count_events()

        stats = {
            "collection_name": vector_store.collection_name,
            "total_events": count,
            "persist_directory": vector_store.persist_directory
        }

        return json.dumps(stats, indent=2)

    except Exception as e:
        return f"Error getting vector store stats: {str(e)}"
