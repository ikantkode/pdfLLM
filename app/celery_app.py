from celery import Celery
from app.config import settings
from app.utils.ocr_processor import OCRProcessor
from app.utils.text_processor import TextProcessor
from app.utils.qdrant_handler import QdrantHandler
import psycopg2
import uuid
import logging
from pathlib import Path
from app.converters import image_converter, doc_converter, excel_converter, txt_converter
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(settings.data_dir) / "logs" / "celery.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "rag_app",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3000
)

# Initialize processors
ocr_processor = OCRProcessor()
text_processor = TextProcessor()
qdrant_handler = QdrantHandler()

def get_db_connection():
    return psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password
    )

def get_file_converter(file_ext: str):
    if file_ext in settings.supported_extensions['images']:
        return image_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['documents']:
        return doc_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['spreadsheets']:
        return excel_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['text']:
        return txt_converter.convert_to_markdown
    return None

def classify_document(content: str) -> str:
    keywords = {
        "submittals": ["ASTM", "submittal", "material", "compliance"],
        "payrolls": ["gross pay", "net pay", "employee", "hours", "rate"],
        "bank_statements": ["deposit", "withdrawal", "balance", "account number"]
    }
    content_lower = content.lower()
    for category, terms in keywords.items():
        if any(term in content_lower for term in terms):
            return category
    return None

@celery_app.task(name="app.celery_app.process_ocr", bind=True, max_retries=3)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def process_ocr(self, file_id: str, user_id: str, file_ext: str, category: str | None, filename: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM file_metadata WHERE file_id = %s AND user_id = %s",
                (file_id, user_id)
            )
            row = cur.fetchone()
            if not row:
                raise Exception("File not found in database")

        file_content = row[0]
        temp_path = Path(settings.temp_upload_dir) / f"{file_id}{file_ext}"
        try:
            temp_path.write_bytes(file_content)

            if file_ext in settings.supported_extensions['pdfs']:
                markdown_content = ocr_processor.process_pdf(str(temp_path))
            else:
                converter = get_file_converter(file_ext)
                if not converter:
                    raise Exception(f"Unsupported file format: {file_ext}")
                markdown_content = converter(str(temp_path))

            is_ocr_likely = file_ext in settings.supported_extensions['images'] or file_ext in settings.supported_extensions['pdfs']
            if is_ocr_likely:
                cleaned_markdown = text_processor.preprocess_ocr_text(markdown_content)
            else:
                cleaned_markdown = text_processor.clean_markdown(markdown_content)

            auto_category = classify_document(cleaned_markdown) if not category else category

            chunk_embeddings = text_processor.generate_embeddings(cleaned_markdown)
            if not chunk_embeddings:
                raise Exception("No embeddings generated for the document")

            for i, (chunk_text, embedding) in enumerate(chunk_embeddings):
                chunk_id = str(uuid.uuid5(uuid.UUID(file_id), f"chunk_{i}"))
                entities = text_processor.extract_entities(chunk_text)
                relationships = text_processor.extract_relationships({"content": chunk_text, "chunk_index": i})

                qdrant_handler.store_chunk(
                    document_id=file_id,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    embedding=embedding,
                    metadata={
                        "filename": filename,
                        "user_id": user_id,
                        "chunk_index": i,
                        "parent_section": text_processor._extract_section(chunk_text),
                        "entities": entities,
                        "relationships": relationships,
                        "category": auto_category
                    }
                )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE file_metadata
                    SET markdown_content = %s, category = %s, status = %s
                    WHERE file_id = %s AND user_id = %s
                    """,
                    (cleaned_markdown, auto_category, "processed", file_id, user_id)
                )
                conn.commit()
            logger.info(f"Processed file: {filename} (ID: {file_id}) with {len(chunk_embeddings)} chunks")
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except Exception as e:
        logger.error(f"OCR processing failed for {filename}: {str(e)}")
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE file_metadata SET status = %s WHERE file_id = %s AND user_id = %s",
                ("failed", file_id, user_id)
            )
            conn.commit()
        raise
    finally:
        conn.close()