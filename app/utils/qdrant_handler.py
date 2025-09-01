import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from app.config import settings

logger = logging.getLogger(__name__)

class QdrantHandler:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection
        self._initialize_collection()

    def _initialize_collection(self):
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            if any(c.name == self.collection_name for c in collections.collections):
                logger.info(f"Collection {self.collection_name} already exists")
                return
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,  # Adjust based on embedding model
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection: {self.collection_name}")
            # Create indexes for metadata fields
            for field in ["document_id", "user_id", "category"]:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_type="keyword"
                )
            logger.info("Created payload indexes")
        except Exception as e:
            if "409" in str(e) or "already exists" in str(e):
                logger.info(f"Collection {self.collection_name} already exists, skipping creation")
            else:
                logger.error(f"Failed to initialize collection: {str(e)}")
                raise

    def store_chunk(self, document_id: str, chunk_id: str, chunk_text: str, embedding: list, metadata: dict):
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": chunk_id,
                        "vector": embedding,
                        "payload": {
                            "document_id": document_id,
                            "content": chunk_text,
                            **metadata
                        }
                    }
                ]
            )
            logger.info(f"Stored chunk {chunk_id} for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk_id}: {str(e)}")
            raise

    def delete_by_document_id(self, document_id: str):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}}
            )
            logger.info(f"Deleted chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {str(e)}")
            raise

    def update_metadata(self, document_id: str, metadata: dict):
        try:
            self.client.update_collection(
                collection_name=self.collection_name,
                points_selector={"filter": {"must": [{"key": "document_id", "match": {"value": document_id}}]}},
                payload=metadata
            )
            logger.info(f"Updated metadata for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update metadata for document {document_id}: {str(e)}")
            raise