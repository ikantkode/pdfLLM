import logging
from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from config import settings

logger = logging.getLogger(__name__)

class QdrantHandler:
    def __init__(self, host: str, port: int, collection_name: str):
        try:
            self.client = QdrantClient(host=host, port=port)
            self.collection_name = collection_name
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
                    vectors_config={"size": 1536, "distance": "Cosine"}
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
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise

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
                # If using fileid_chunkindex format, convert to UUID
                base_uuid = chunk_id.split("_")[0]
                try:
                    # Try to use the base UUID portion
                    point_id = uuid.UUID(base_uuid)
                except ValueError:
                    # If not valid UUID, generate a new one from the string
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
            else:
                # If already a UUID, use it directly
                point_id = uuid.UUID(chunk_id)

            payload = {
                "document_id": document_id,
                "content": chunk_text,
                **metadata
            }

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": str(point_id),  # Ensure string representation of UUID
                        "vector": embedding,
                        "payload": payload
                    }
                ],
                wait=True
            )
            logger.info(f"Stored chunk {chunk_id} for document {document_id}")
        except Exception as e:
            logger.error(f"Failed to store chunk {chunk_id}: {str(e)}")
            raise

    async def save_chunk(self, chunk: Dict, user_id: str):
        """Save a chunk to Qdrant"""
        try:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=chunk['embedding'],
                payload={
                    "user_id": user_id,
                    "document_id": chunk['document_id'],
                    "content": chunk['content'],
                    "chunk_index": chunk['chunk_index'],
                    "parent_section": chunk['parent_section'],
                    "entities": chunk['entities'],
                    "relationships": chunk['relationships']
                }
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            logger.debug(f"Saved chunk {chunk['chunk_index']} for document {chunk['document_id']}")
        except Exception as e:
            logger.error(f"Failed to save chunk: {str(e)}")
            raise

    async def search_entities(self, entities: List[str], user_id: str, file_ids: Optional[List[str]] = None, limit: int = 5) -> List[Any]:
        """Search for chunks containing specific entities"""
        try:
            results = []
            for entity in entities:
                filter_conditions = [
                    FieldCondition(
                        key="entities",
                        match=MatchValue(value=entity)
                    ),
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
                if file_ids:
                    filter_conditions.append(
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=file_ids, match_any=True)
                        )
                    )

                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_filter=Filter(must=filter_conditions),
                    limit=limit,
                    with_vectors=False
                )
                results.extend(search_results)
            return results
        except Exception as e:
            logger.error(f"Entity search failed: {str(e)}")
            return []

    async def delete_by_document_id(self, document_id: str):
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
