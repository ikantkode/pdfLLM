from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantHandler:
    def __init__(self, host="qdrant", port=6333, collection_name="rag_chunks"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        try:
            self.create_collection()
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {str(e)}")
            raise

    def create_collection(self):
        try:
            collections = self.client.get_collections()
            if self.collection_name not in [c.name for c in collections.collections]:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # Dimension for text-embedding-3-small
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {str(e)}")
            raise

    def save_chunk(self, chunk_data, user_id):
        try:
            point = PointStruct(
                id=chunk_data['chunk_id'],
                vector=chunk_data['embedding'],
                payload={
                    "document_id": chunk_data['document_id'],
                    "content": chunk_data['content'],
                    "parent_section": chunk_data['parent_section'],
                    "chunk_index": chunk_data['chunk_index'],
                    "user_id": user_id
                }
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"Saved chunk {chunk_data['chunk_id']} to Qdrant")
        except Exception as e:
            logger.error(f"Failed to save chunk {chunk_data['chunk_id']}: {str(e)}")
            raise

    def delete_by_document_id(self, document_id):
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                )
            )
            logger.info(f"Deleted chunks for document_id: {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunks for document_id {document_id}: {str(e)}")
            raise