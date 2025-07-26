import asyncio
from celery import Celery
from app.config import settings
from app.utils.ocr_processor import OCRProcessor
from app.utils.text_processor import TextProcessor
from app.utils.qdrant_handler import QdrantHandler
import psycopg2
import uuid
import logging
import os
from pathlib import Path
from app.converters import image_converter, doc_converter, excel_converter, txt_converter
from tenacity import retry, stop_after_attempt, wait_exponential
from contextlib import contextmanager

# Configure logging
Path(settings.data_dir).joinpath("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(task_id)s - %(message)s',
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
try:
    ocr_processor = OCRProcessor()
    text_processor = TextProcessor()
    qdrant_handler = QdrantHandler()
    logger.info("Processors initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize processors: {str(e)}")
    raise

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password
        )
        logger.debug("PostgreSQL connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
        raise

@contextmanager
def temp_file_handler(temp_path: Path):
    try:
        yield
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Failed to clean up temporary file {temp_path}: {str(e)}")

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

def has_last_error_column(conn):
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'file_metadata' AND column_name = 'last_error'
                """
            )
            return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Failed to check for last_error column: {str(e)}")
        return False

@celery_app.task(name="app.celery_app.process_ocr", bind=True, max_retries=3)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
def process_ocr(self, file_id: str, user_id: str, file_ext: str, category: str | None, filename: str):
    task_id = self.request.id or "unknown"
    logger.info(f"Starting OCR processing for file_id: {file_id}, filename: {filename}, task_id: {task_id}", extra={"task_id": task_id})

    conn = get_db_connection()
    try:
        # Validate file metadata
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content FROM file_metadata WHERE file_id = %s AND user_id = %s",
                (file_id, user_id)
            )
            row = cur.fetchone()
            if not row:
                logger.error(f"File not found in database: file_id={file_id}, user_id={user_id}", extra={"task_id": task_id})
                raise Exception("File not found in database")
            file_content = row[0]
            if not file_content:
                logger.error(f"Empty file content for file_id={file_id}, filename={filename}", extra={"task_id": task_id})
                raise Exception("Empty file content")

        # Validate temp_upload_dir permissions
        temp_dir = Path(settings.temp_upload_dir)
        if not os.access(temp_dir, os.W_OK):
            logger.error(f"Temp directory {temp_dir} is not writable", extra={"task_id": task_id})
            raise Exception(f"Temp directory {temp_dir} is not writable")

        # Write temporary file
        temp_path = temp_dir / f"{file_id}{file_ext}"
        with temp_file_handler(temp_path):
            try:
                temp_path.write_bytes(file_content)
                logger.debug(f"Wrote temporary file: {temp_path}", extra={"task_id": task_id})
            except Exception as e:
                logger.error(f"Failed to write temporary file {temp_path}: {str(e)}", extra={"task_id": task_id})
                raise Exception(f"Failed to write temporary file: {str(e)}")

            # Process file
            try:
                if file_ext in settings.supported_extensions['pdfs']:
                    markdown_content = ocr_processor.process_pdf(str(temp_path))
                else:
                    converter = get_file_converter(file_ext)
                    if not converter:
                        logger.error(f"Unsupported file format: {file_ext} for file_id={file_id}", extra={"task_id": task_id})
                        raise Exception(f"Unsupported file type: {file_ext}")
                    markdown_content = converter(str(temp_path))
                logger.debug(f"Converted file {filename} to markdown", extra={"task_id": task_id})
            except Exception as e:
                logger.error(f"File conversion failed for {filename}: {str(e)}", extra={"task_id": task_id})
                raise Exception(f"File conversion failed: {str(e)}")

            # Preprocess markdown
            try:
                is_ocr_likely = file_ext in settings.supported_extensions['images'] or file_ext in settings.supported_extensions['pdfs']
                if is_ocr_likely:
                    logger.warning(f"Using clean_markdown as fallback for OCR preprocessing for {filename}, as preprocess_ocr_text is not available", extra={"task_id": task_id})
                    cleaned_markdown = text_processor.clean_markdown(markdown_content)
                else:
                    cleaned_markdown = text_processor.clean_markdown(markdown_content)
                logger.debug(f"Preprocessed markdown for {filename}", extra={"task_id": task_id})
            except Exception as e:
                logger.error(f"Markdown preprocessing failed for {filename}: {str(e)}", extra={"task_id": task_id})
                raise Exception(f"Markdown preprocessing failed: {str(e)}")

            # Classify document
            auto_category = classify_document(cleaned_markdown) if not category else category

            # Generate embeddings
            try:
                embeddings_coro = text_processor.generate_embeddings(cleaned_markdown)
                chunk_embeddings = asyncio.run(embeddings_coro)
                if not chunk_embeddings:
                    logger.error(f"No embeddings generated for {filename}", extra={"task_id": task_id})
                    raise Exception("No embeddings generated for the document")
                logger.debug(f"Generated {len(chunk_embeddings)} embeddings for {filename}", extra={"task_id": task_id})
            except Exception as e:
                logger.error(f"Embedding generation failed for {filename}: {str(e)}", extra={"task_id": task_id})
                raise Exception(f"Embedding generation failed: {str(e)}")

            # Store chunks in Qdrant
            try:
                for i, (chunk_text, embedding) in enumerate(chunk_embeddings):
                    chunk_id = str(uuid.uuid5(uuid.UUID(file_id), f"chunk_{i}"))
                    entities_coro = text_processor.extract_entities(chunk_text)
                    entities = asyncio.run(entities_coro)
                    relationships_coro = text_processor.extract_relationships({"content": chunk_text, "chunk_index": i})
                    relationships = asyncio.run(relationships_coro)

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
                logger.debug(f"Stored {len(chunk_embeddings)} chunks in Qdrant for {filename}", extra={"task_id": task_id})
            except Exception as e:
                logger.error(f"Qdrant storage failed for {filename}: {str(e)}", extra={"task_id": task_id})
                raise Exception(f"Qdrant storage failed: {str(e)}")

            # Update database
            try:
                with conn.cursor() as cur:
                    if has_last_error_column(conn):
                        cur.execute(
                            """
                            UPDATE file_metadata
                            SET markdown_content = %s, category = %s, status = %s, last_error = %s
                            WHERE file_id = %s AND user_id = %s
                            """,
                            (cleaned_markdown, auto_category, "processed", None, file_id, user_id)
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE file_metadata
                            SET markdown_content = %s, category = %s, status = %s
                            WHERE file_id = %s AND user_id = %s
                            """,
                            (cleaned_markdown, auto_category, "processed", file_id, user_id)
                        )
                    if cur.rowcount == 0:
                        logger.error(f"No rows updated in file_metadata for file_id: {file_id}", extra={"task_id": task_id})
                        raise Exception("Failed to update metadata: No rows affected")
                    conn.commit()
                    logger.info(f"Processed file: {filename} (ID: {file_id}) with {len(chunk_embeddings)} chunks", extra={"task_id": task_id})
            except Exception as e:
                logger.error(f"Database update failed for {filename}: {str(e)}", extra={"task_id": task_id})
                raise Exception(f"Database update failed: {str(e)}")

    except Exception as e:
        logger.error(f"OCR processing failed for {filename} (ID: {file_id}): {str(e)}", extra={"task_id": task_id})
        try:
            with conn.cursor() as cur:
                if has_last_error_column(conn):
                    cur.execute(
                        """
                        UPDATE file_metadata
                        SET status = %s, last_error = %s
                        WHERE file_id = %s AND user_id = %s
                        """,
                        ("failed", str(e), file_id, user_id)
                    )
                else:
                    cur.execute(
                        """
                        UPDATE file_metadata
                        SET status = %s
                        WHERE file_id = %s AND user_id = %s
                        """,
                        ("failed", file_id, user_id)
                    )
                conn.commit()
                logger.debug(f"Updated status to failed for file_id: {file_id}", extra={"task_id": task_id})
        except Exception as db_e:
            logger.error(f"Failed to update status to failed for {filename}: {str(db_e)}", extra={"task_id": task_id})
        raise
    finally:
        conn.close()