import logging
from typing import List, Dict, Any, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue

from app.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class QdrantHandler:
    def __init__(self):
        try:
            self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=30)
            self.collection_name = "pdfllm_collection"
            self.vector_size = 1024
            self._initialize_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

    def _initialize_collection(self):
        """Initialize Qdrant collection with vector and payload indexes"""
        try:
            collections = self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")

                # Create payload indexes for efficient filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="user_id",
                    field_schema="keyword"
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema="keyword"
                )
                logger.info("Created payload indexes")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def store_chunk(
        self,
        document_id: str,
        chunk_id: str,
        chunk_text: str,
        embedding: List[float],
        metadata: Dict
    ):
        """Store a document chunk in Qdrant"""
        try:
            # Convert chunk_id to valid UUID format
            if "_" in chunk_id:
                base_uuid = chunk_id.split("_")[0]
                try:
                    point_id = uuid.UUID(base_uuid)
                except ValueError:
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
            else:
                point_id = uuid.UUID(chunk_id)

            # Validate metadata to ensure it's serializable
            if not isinstance(metadata.get("entities", []), list) or not all(isinstance(e, (str, dict)) for e in metadata["entities"]):
                raise ValueError("Entities must be a list of strings or dicts")
            if not isinstance(metadata.get("relationships", []), list) or not all(isinstance(r, dict) for r in metadata["relationships"]):
                raise ValueError("Relationships must be a list of dicts")

            payload = {
                "document_id": document_id,
                "content": chunk_text,
                **metadata
            }

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(point_id),
                        vector=embedding,
                        payload=payload
                    )
                ],
                wait=True
            )
            logger.info(f"Stored chunk {chunk_id} for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk_id}: {str(e)}")
            raise

    def delete_by_document_id(self, document_id: str):
        """Delete all chunks associated with a document ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True
            )
            logger.info(f"Deleted all chunks for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunks for document {document_id}: {str(e)}")
            raise

    def update_metadata(self, document_id: str, new_metadata: dict):
        """Update metadata for all chunks of a document"""
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                with_payload=True
            )
            for point in points:
                updated_payload = point.payload.copy()
                updated_payload.update(new_metadata)
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=updated_payload,
                    points=[point.id],
                    wait=True
                )
            logger.info(f"Updated metadata for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to update metadata for document {document_id}: {str(e)}")
            raise