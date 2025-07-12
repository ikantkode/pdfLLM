import os
import uuid
import json
import logging
import base64
import datetime
import hashlib
import re
from typing import List, Dict, Optional, Any
from io import BytesIO
from pathlib import Path
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    status,
    Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.encoders import jsonable_encoder
import tiktoken
import networkx as nx
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel
from config import settings
from utils.qdrant_handler import QdrantHandler
from utils.text_processor import TextProcessor
from utils.ocr_processor import OCRProcessor
from converters import (
    image_converter,
    doc_converter,
    excel_converter,
    txt_converter
)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Configure logging
Path(settings.data_dir).joinpath("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(settings.data_dir) / "logs" / "rag_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A Retrieval-Augmented Generation (RAG) microservice",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key")

async def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Initialize OCR processor
ocr_processor = OCRProcessor()

# Initialize Qdrant with retry logic
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=20),
    reraise=True
)
def initialize_qdrant_handler():
    try:
        qdrant_handler = QdrantHandler(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection
        )
        logger.info("Qdrant connection established")
        return qdrant_handler
    except Exception as e:
        logger.error(f"Qdrant connection attempt failed: {str(e)}")
        raise

try:
    qdrant_handler = initialize_qdrant_handler()
except Exception as e:
    logger.error(f"Qdrant connection failed after retries: {str(e)}")
    raise

try:
    text_processor = TextProcessor()
    logger.info("TextProcessor initialized successfully")
except Exception as e:
    logger.error(f"TextProcessor initialization failed: {str(e)}")
    raise

# Create necessary directories
Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)

# Models
class FileMetadata(BaseModel):
    file_id: str
    filename: str
    file_type: str
    upload_date: str
    content: str
    markdown_content: str
    user_id: str
    size: int
    checksum: str

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class ChatSession(BaseModel):
    chat_id: str
    user_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
    document_ids: List[str]

class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    parent_section: str
    chunk_index: int
    content: str
    entities: List[str]
    relationships: List[Dict[str, str]]
    score: float

# Knowledge Graph for advanced indexing
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_index = {}

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship: str,
        weight: float = 1.0
    ) -> None:
        """Add a relationship between entities"""
        logger.debug(f"Adding relationship: {source} -> {relationship} -> {target}")
        if source not in self.graph:
            self.graph.add_node(source, type='entity')
        if target not in self.graph:
            self.graph.add_node(target, type='entity')
        self.graph.add_edge(
            source,
            target,
            relationship=relationship,
            weight=weight
        )

    def find_related_entities(
        self,
        entity: str,
        depth: int = settings.max_graph_depth
    ) -> List[str]:
        """Find related entities within specified depth"""
        if entity not in self.graph:
            return []
        return list(nx.single_source_shortest_path_length(
            self.graph,
            entity,
            cutoff=depth
        ).keys())

    def save(self) -> None:
        """Save graph to file"""
        data = nx.node_link_data(self.graph)
        try:
            with open(settings.graph_file, 'w') as f:
                json.dump(data, f)
            logger.info("Knowledge graph saved successfully")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {str(e)}")

    def load(self) -> None:
        """Load graph from file"""
        if os.path.exists(settings.graph_file):
            try:
                with open(settings.graph_file, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                logger.info("Knowledge graph loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load knowledge graph: {str(e)}")

knowledge_graph = KnowledgeGraph()

# State management
class StateManager:
    def __init__(self):
        self.state_file = Path(settings.data_dir) / "state.json"
        self.file_metadata: List[Dict[str, Any]] = []
        self.chat_sessions: Dict[str, Dict[str, Any]] = {}
        self.load()

    def save(self) -> None:
        """Save application state"""
        state = {
            "file_metadata": self.file_metadata,
            "chat_sessions": self.chat_sessions
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(jsonable_encoder(state), f)
            knowledge_graph.save()
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def load(self) -> None:
        """Load application state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    self.file_metadata = state.get("file_metadata", [])
                    self.chat_sessions = state.get("chat_sessions", {})
                knowledge_graph.load()
                logger.info("State loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load state: {str(e)}")

state_manager = StateManager()

# Helper functions
def get_file_converter(file_ext: str):
    """Get the appropriate converter for a file extension"""
    if file_ext in settings.supported_extensions['images']:
        return image_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['documents']:
        return doc_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['spreadsheets']:
        return excel_converter.convert_to_markdown
    elif file_ext in settings.supported_extensions['text']:
        return txt_converter.convert_to_markdown
    return None

def split_large_text(text: str, max_tokens: int = settings.max_embedding_tokens) -> List[str]:
    """Split text into chunks, preserving table structures and semantic units"""
    # Detect markdown tables
    table_pattern = r'(\|.*?\|\n(?:\|[-: ]+\|\n)+.*?)(?=\n\n|\Z)'
    tables = re.findall(table_pattern, text, re.DOTALL)
    non_table_text = re.sub(table_pattern, 'TABLE_PLACEHOLDER', text)

    chunks = []
    current_chunk = ""
    token_count = 0
    placeholder_count = 0

    # Split non-table text by sentences
    doc = text_processor.nlp(non_table_text)
    for sent in doc.sents:
        sent_text = sent.text
        if 'TABLE_PLACEHOLDER' in sent_text:
            # Replace placeholder with actual table
            if placeholder_count < len(tables):
                sent_text = sent_text.replace('TABLE_PLACEHOLDER', tables[placeholder_count])
                placeholder_count += 1
        sent_tokens = len(tokenizer.encode(sent_text))

        if token_count + sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                token_count = 0
            if sent_tokens > max_tokens:
                # Split large sentence further
                sub_chunks = [sent_text[i:i + max_tokens] for i in range(0, len(sent_text), max_tokens)]
                chunks.extend(sub_chunks)
                continue

        current_chunk += sent_text + "\n"
        token_count += sent_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add remaining tables
    while placeholder_count < len(tables):
        table_tokens = len(tokenizer.encode(tables[placeholder_count]))
        if table_tokens <= max_tokens:
            chunks.append(tables[placeholder_count])
        else:
            # Split large table into rows
            rows = tables[placeholder_count].split('\n')
            sub_chunk = ""
            sub_tokens = 0
            for row in rows:
                row_tokens = len(tokenizer.encode(row))
                if sub_tokens + row_tokens > max_tokens:
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                        sub_chunk = ""
                        sub_tokens = 0
                    sub_chunk += row + "\n"
                    sub_tokens += row_tokens
                else:
                    sub_chunk += row + "\n"
                    sub_tokens += row_tokens
            if sub_chunk:
                chunks.append(sub_chunk.strip())
        placeholder_count += 1

    logger.info(f"Split text into {len(chunks)} chunks with max {max_tokens} tokens")
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings with batch processing and retry logic"""
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def preprocess_ocr_text(text: str) -> str:
    """Use LLM to clean and reconstruct OCR-extracted text into coherent sentences."""
    # Split large text to avoid token limits, preserving tables
    chunks = split_large_text(text, max_tokens=settings.max_embedding_tokens // 2)
    cleaned_chunks = []

    client = OpenAI(api_key=settings.openai_api_key)
    for chunk in chunks:
        try:
            prompt = (
                        f"Input Text:\n{chunk}\n\n"
            "You are an intelligent document cleaner. The input text may contain OCR errors such as:\n"
            "- Extra spaces between characters (e.g., 'D z i u b a', '5 0 . 4 3')\n"
            "- Malformed numbers (e.g., '1 , 98 . 16', '$$50.43', '7,068.92Net5,079.50')\n"
            "- Repeated words, misaligned formatting, or broken names (e.g., 'E r m i l o L o p e z')\n"
            "- Incomplete, gibberish, or disorganized text\n\n"

            "Your tasks are:\n"
            "1. **Reconstruct** the text into clear, grammatically correct, and logically structured content.\n"
            "2. **Fix character spacing** in names, words, and numbers.\n"
            "3. **Standardize numbers and currency** formatting (e.g., '1 , 98 . 16' → '$1,988.16'). Use a single `$` where needed.\n"
            "4. **Rejoin broken names or words** (e.g., 'E r m i l o L o p e z' → 'Ermilo Lopez').\n"
            "5. **Preserve markdown** formatting such as headings, lists, and tables. Ensure tables are well-aligned.\n"
            "6. **If the input contains payroll-related information** (such as employee names, hours, pay rates, gross/net pay):\n"
            "   - Extract the relevant data and structure it in a markdown table.\n"
            "   - Columns should include: `Employee`, `Role`, `Hours`, `Rate`, `Gross Pay`, `Net Pay`.\n"
            "   - Ensure Gross Pay = Hours × Rate, and Net Pay < Gross Pay. Make reasonable assumptions and note them if necessary.\n"
            "7. **If the input is not payroll-related**, simply clean and format the content into readable markdown, preserving any headings, paragraphs, or lists.\n"
            "8. **Remove noise or gibberish**, but preserve all meaningful data.\n"
            "9. **Comment assumptions** using HTML comments (e.g., `<!-- Assumed currency as USD -->`).\n"
            "10. **Return only the cleaned text in markdown format**.\n\n"

            "Example Input (Payroll):\n"
            "Z e n o v i i D z i u b a L a b o r e r 1 6 h o u r s 9 4 . 4 3 p e r h o u r 1 , 5 1 0 . 8 8 N e t 1 , 1 4 3 . 2 0\n"
            "E r m i l o L o p e z L a b o r e r 8 h o u r s 9 4 . 4 3 p e r h o u r 7 5 5 . 4 4 N e t 6 0 8 . 6 5\n\n"
            
            "Example Output (Payroll):\n"
            "```markdown\n"
            "## Payroll Details\n"
            "| Employee        | Role     | Hours | Rate             | Gross Pay | Net Pay  |\n"
            "|-----------------|----------|-------|------------------|-----------|----------|\n"
            "| Zenovii Dziuba  | Laborer  | 16    | $94.43 per hour  | $1,510.88 | $1,143.20|\n"
            "| Ermilo Lopez    | Laborer  | 8     | $94.43 per hour  | $755.44   | $608.65  |\n"
            "<!-- Assumed currency as USD -->\n"
            "```\n\n"

            "Example Input (Non-payroll):\n"
            "T h e   p r o j e c t   w a s   i n i t i a t e d   o n   0 5 . 0 6 . 2 0 2 3   a n d   i n v o l v e d   r e g u l a r   s a f e t y   i n s p e c t i o n s .\n"

            "Example Output (Non-payroll):\n"
            "```markdown\n"
            "The project was initiated on 05.06.2023 and involved regular safety inspections.\n"
            "```\n"
            )

            response = client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at cleaning and reconstructing noisy OCR-extracted text. "
                            "Correct OCR errors, including spaces between characters in words, names, or numbers. "
                            "Format numbers and names properly, ensuring single '$' for currency, and structure the output in clear, coherent markdown. "
                            "Preserve meaningful information and structural elements (e.g., lists, tables). "
                            "For payroll data, format details in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay. "
                            "Validate numerical consistency: Gross Pay = Hours × Rate, Net Pay < Gross Pay. Use context-provided rates (e.g., $94.43 per hour) if available. "
                            "Note any assumptions made due to ambiguous text in markdown comments."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.max_completion_tokens,
                temperature=0.3
            )
            cleaned_chunk = response.choices[0].message.content.strip()
            cleaned_chunks.append(cleaned_chunk)
            logger.debug(f"Preprocessed chunk: {cleaned_chunk[:200]}...")
        except Exception as e:
            logger.error(f"Failed to preprocess OCR chunk: {str(e)}")
            raise

    # Combine cleaned chunks
    cleaned_text = "\n\n".join(cleaned_chunks)
    logger.debug(f"Preprocessed OCR text: {cleaned_text[:500]}...")
    return cleaned_text

def clean_response(text: str) -> str:
    """Clean the LLM response to fix residual OCR artifacts and table formatting"""
    logger.debug(f"Raw response before cleaning: {text[:500]}...")

    # Fix spaces between characters in words, names, or numbers, but preserve spaces between words
    text = re.sub(r'(\w)\s{2,}(\w)', r'\1 \2', text)  # Fix multiple spaces between words
    text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)  # Fix spaced numbers (e.g., "1 , 510 . 88" → "1,510.88")
    text = re.sub(r'(\w)\s+([.,:;])\s+(\w)', r'\1\2\3', text)  # Fix spaced punctuation (e.g., "C P R . p d f" → "CPR.pdf")
    text = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4', text)  # Fix spaced words (e.g., "N e t P a y" → "NetPay")
    text = re.sub(r'\${2,}', r'$', text)  # Fix double dollar signs (e.g., "$$94.43" → "$94.43")

    # Fix payroll-specific formats
    text = re.sub(r'(\d+\.\d{2})\s*N\s*e\s*t\s*(\d+\.\d{2})', r'Gross $\1, Net $\2', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+\.\d{2})\s*p\s*e\s*r\s*h\s*o\s*u\s*r', r'$\1 per hour', text, flags=re.IGNORECASE)

    # Validate and correct table numerical consistency
    lines = text.split('\n')
    cleaned_lines = []
    in_table = False
    table_lines = []
    total_gross = 0.0
    total_net = 0.0

    for line in lines:
        if line.strip().startswith('|') and not in_table:
            in_table = True
            table_lines = [line]
        elif in_table and line.strip().startswith('|'):
            table_lines.append(line)
        elif in_table and not line.strip().startswith('|'):
            in_table = False
            # Clean and validate table
            cleaned_table = []
            header = None
            net_pay_values = []
            for table_line in table_lines:
                cells = [re.sub(r'(\w)\s{2,}(\w)', r'\1 \2', cell.strip()) for cell in table_line.split('|')]
                cells = [re.sub(r'\${2,}', r'$', cell) for cell in cells]  # Fix $$ in table cells
                if not header and "Employee" in cells:
                    header = cells
                    cleaned_table.append('|'.join(cells))
                    continue
                if header and len(cells) >= len(header):
                    try:
                        hours = float(cells[header.index('Hours')]) if 'Hours' in header else 0.0
                        rate_str = cells[header.index('Rate')] if 'Rate' in header else ''
                        gross_str = cells[header.index('Gross Pay')] if 'Gross Pay' in header else ''
                        net_str = cells[header.index('Net Pay')] if 'Net Pay' in header else ''
                        rate = float(re.search(r'\d+\.\d{2}', rate_str).group()) if rate_str and re.search(r'\d+\.\d{2}', rate_str) else 0.0
                        gross = float(re.search(r'\d+\.\d{2}', gross_str).group()) if gross_str and re.search(r'\d+\.\d{2}', gross_str) else 0.0
                        net = float(re.search(r'\d+\.\d{2}', net_str).group()) if net_str and re.search(r'\d+\.\d{2}', net_str) else 0.0
                        # Validate: Gross Pay = Hours × Rate, Net Pay < Gross Pay
                        expected_gross = hours * rate
                        if abs(expected_gross - gross) > 0.01:
                            logger.warning(f"Inconsistent gross pay: {gross} != {hours} × {rate} = {expected_gross}")
                            gross = expected_gross
                            cells[header.index('Gross Pay')] = f'${gross:.2f}'
                        if net > gross:
                            logger.warning(f"Net pay {net} exceeds gross pay {gross}, swapping values")
                            cells[header.index('Net Pay')], cells[header.index('Gross Pay')] = cells[header.index('Gross Pay')], cells[header.index('Net Pay')]
                            net, gross = gross, net
                        total_gross += gross
                        net_pay_values.append(net)
                        total_net = sum(net_pay_values)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to validate table row: {cells}, error: {str(e)}")
                    cleaned_table.append('|'.join(cells))
            cleaned_lines.extend(cleaned_table)
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    if in_table:
        # Clean remaining table
        cleaned_table = []
        header = None
        net_pay_values = []
        for table_line in table_lines:
            cells = [re.sub(r'(\w)\s{2,}(\w)', r'\1 \2', cell.strip()) for cell in table_line.split('|')]
            cells = [re.sub(r'\${2,}', r'$', cell) for cell in cells]  # Fix $$ in table cells
            if not header and "Employee" in cells:
                header = cells
                cleaned_table.append('|'.join(cells))
                continue
            if header and len(cells) >= len(header):
                try:
                    hours = float(cells[header.index('Hours')]) if 'Hours' in header else 0.0
                    rate_str = cells[header.index('Rate')] if 'Rate' in header else ''
                    gross_str = cells[header.index('Gross Pay')] if 'Gross Pay' in header else ''
                    net_str = cells[header.index('Net Pay')] if 'Net Pay' in header else ''
                    rate = float(re.search(r'\d+\.\d{2}', rate_str).group()) if rate_str and re.search(r'\d+\.\d{2}', rate_str) else 0.0
                    gross = float(re.search(r'\d+\.\d{2}', gross_str).group()) if gross_str and re.search(r'\d+\.\d{2}', gross_str) else 0.0
                    net = float(re.search(r'\d+\.\d{2}', net_str).group()) if net_str and re.search(r'\d+\.\d{2}', net_str) else 0.0
                    expected_gross = hours * rate
                    if abs(expected_gross - gross) > 0.01:
                        gross = expected_gross
                        cells[header.index('Gross Pay')] = f'${gross:.2f}'
                    if net > gross:
                        cells[header.index('Net Pay')], cells[header.index('Gross Pay')] = cells[header.index('Gross Pay')], cells[header.index('Net Pay')]
                        net, gross = gross, net
                    total_gross += gross
                    net_pay_values.append(net)
                    total_net = sum(net_pay_values)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to validate table row: {cells}, error: {str(e)}")
            cleaned_table.append('|'.join(cells))
        cleaned_lines.extend(cleaned_table)

    # Normalize whitespace
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n[ \t]+', '\n', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = text.strip()

    # Fix spaced characters in total expenditure line
    text = re.sub(r'G r o s s\s+P a y', 'Gross Pay', text)
    text = re.sub(r'N e t\s+P a y', 'Net Pay', text)
    text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)  # Fix spaced numbers in totals
    text = re.sub(r'\${2,}', r'$', text)  # Fix double dollar signs in totals

    # Append corrected totals only if a table was processed and totals are present
    if total_gross > 0 and total_net > 0:
        text = re.sub(
            r'Total expenditure:.*$',
            f'Total expenditure: Gross Pay ${total_gross:.2f}, Net Pay ${total_net:.2f}',
            text,
            flags=re.MULTILINE
        )

    logger.debug(f"Cleaned response: {text[:500]}...")
    return text

def format_search_results(results: List) -> List[SearchResult]:
    """Format search results for response"""
    file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}
    return [
        SearchResult(
            chunk_id=str(r.id),
            document_id=r.payload.get("document_id", "N/A"),
            filename=file_map.get(r.payload.get("document_id", "N/A"), "Unknown"),
            parent_section=r.payload.get("parent_section", "N/A"),
            chunk_index=r.payload.get("chunk_index", 0),
            content=r.payload.get("content", "N/A"),
            entities=r.payload.get("entities", []),
            relationships=r.payload.get("relationships", []),
            score=r.score
        )
        for r in results
    ]

def rank_results(entity_results: List, vector_results: List, limit: int) -> List:
    """Combine and rank results from different retrieval methods"""
    seen = set()
    combined = []

    for result in entity_results + vector_results:
        result_id = result.id if hasattr(result, 'id') else result.get('chunk_id')
        if result_id and result_id not in seen:
            seen.add(result_id)
            score = (result.score if hasattr(result, 'score') else result.get('score', 0.0))
            score += 0.1 * len(result.payload.get('entities', []))  # Boost for entities
            score += 0.2 * len(result.payload.get('relationships', []))  # Boost for relationships
            result.score = score
            combined.append(result)

    return sorted(
        combined,
        key=lambda x: x.score if hasattr(x, 'score') else x.get('score', 0.0),
        reverse=True
    )[:limit]

def clean_query(query: str) -> str:
    """Clean user query to handle OCR-like noise and payroll-specific formatting"""
    # Normalize whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Remove excessive special characters
    query = re.sub(r'([^\w\s])\1+', r'\1', query)
    
    # Remove repeated words/phrases
    query = re.sub(r'\b(\w+)\s+\1\b', r'\1', query, flags=re.IGNORECASE)
    
    # Remove invalid characters, preserving common punctuation and currency
    query = re.sub(r'[^\w\s\.\,\$\(\)\-]', '', query)
    
    # Fix malformed numbers
    query = re.sub(r'(\d+),(\d{2})\.(\d{2})', r'\1\2.\3', query)
    
    # Correct payroll-specific formats
    query = re.sub(r'(\d+\.\d{2})Net(\d+\.\d{2})', r'Gross $\1, Net $\2', query)
    query = re.sub(r'(\d+\.\d{2})\s*perhour', r'$\1 per hour', query, flags=re.IGNORECASE)
    
    # Fix fragmented names
    query = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4\5', query)
    query = re.sub(r'\b(\w)\s+(\w)\s+(\w)\s+(\w)\b', r'\1\2\3\4', query)
    
    # Fix document names
    query = re.sub(r'CPRPreviewSCA\s*\((\d+)\)\.pdf', r'CPRPreviewSCA_\1.pdf', query)
    
    logger.debug(f"Cleaned query: {query}")
    return query

async def build_chat_context(results: List[SearchResult]) -> str:
    """Build context from search results with token limits"""
    context = ""
    token_count = 0
    file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}
    seen_content = set()

    for result in results:
        content = result.content
        # Re-clean retrieved content to handle any residual noise
        cleaned_content = re.sub(r'(\w)\s{2,}(\w)', r'\1 \2', content)  # Fix multiple spaces
        cleaned_content = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', cleaned_content)  # Fix spaced numbers
        cleaned_content = re.sub(r'\${2,}', r'$', cleaned_content)  # Fix double dollar signs
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()  # Normalize whitespace
        
        # Skip empty or duplicate content
        if not cleaned_content or cleaned_content in seen_content:
            logger.debug(f"Skipped content: {cleaned_content[:100]}...")
            continue
        seen_content.add(cleaned_content)

        tokens = len(tokenizer.encode(cleaned_content))
        if token_count + tokens > settings.max_completion_tokens * 0.8:
            logger.debug(f"Context truncated at {token_count} tokens")
            break

        context += f"Document: {file_map.get(result.document_id, 'Unknown')}\n"
        context += f"Section: {result.chunk_index}\n"
        context += f"Entities: {', '.join(result.entities)}\n"
        context += f"Relationships: {', '.join(['{} {} {}'.format(rel['subject'], rel['predicate'], rel['object']) for rel in result.relationships])}\n"
        context += f"Content: {cleaned_content}\n\n"
        token_count += tokens

    logger.debug(f"Built context with {token_count} tokens: {context[:500]}...")
    return context

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def generate_coherent_response(query: str, context: str) -> str:
    """Generate response with proper token management and assistant-like behavior"""
    client = OpenAI(api_key=settings.openai_api_key)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        "You are a professional personal assistant tasked with providing a clear, concise, and accurate response based solely on the provided context and query. "
        "The context has been preprocessed to minimize OCR errors, but minor issues may remain (e.g., spaces between characters, malformed numbers). "
        "Follow these instructions:\n"
        "1. Structure the response in a well-organized manner with complete sentences and proper grammar.\n"
        "2. Avoid repetition of words, phrases, or numbers unless necessary for clarity.\n"
        "3. Format numerical values correctly, using a single '$' for currency (e.g., '$1,988.16', '$94.43 per hour').\n"
        "4. Remove spaces between characters in words, names, or numbers (e.g., 'D z i u b a' → 'Dziuba', '5 0 . 4 3' → '50.43').\n"
        "5. For tables, ensure proper markdown formatting with aligned columns and no spaces between characters (e.g., '| Zenovii Dziuba | Laborer | 16 | $94.43 per hour | $1,510.88 | $1,143.20 | CPRPreviewSCA_6.pdf (Section 0) |').\n"
        "6. Include a 'Source' column in tables to cite document name and section number for each piece of information.\n"
        "7. Include relevant entities or relationships only if they directly contribute to answering the query.\n"
        "8. Do not speculate or include information not present in the context or query.\n"
        "9. For queries asking to 'explain' or describe a document (e.g., 'what is this document about'), provide a narrative summary of its content, purpose, and key details, followed by a markdown table of payroll details with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay, Source.\n"
        "10. Only include total expenditure (Gross Pay and Net Pay) if the query explicitly asks for it (e.g., contains 'total', 'expenditure', or 'spent').\n"
        "11. Validate numerical consistency: ensure Gross Pay = Hours × Rate, and Net Pay < Gross Pay. Use context-provided rates (e.g., $94.43 per hour) if available.\n"
        "12. Keep the response under {settings.max_completion_tokens} tokens.\n"
        "13. If minor OCR errors remain, prioritize meaningful information and ignore jumbled characters or formatting errors.\n"
        "14. If hours or rates are missing, estimate them using gross pay and typical rates from the context (e.g., $94.43 per hour), noting assumptions.\n"
        "15. If the context or query is unclear, note limitations (e.g., 'Some details may be incomplete due to minor OCR errors').\n\n"
        "Example response for 'What is this document about?':\n"
        """The document 'CPRPreviewSCA_6.pdf' is a certified payroll report from AMB Contractors Inc. for the P.S. 54 - Bronx project, managed by the New York City School Construction Authority and the Department of Education. This report provides detailed information on wages paid to employees for the week ending April 23, 2023. It includes information about the contractor, subcontractor, and employees, along with their roles, hours worked, hourly rates, and corresponding gross and net pay. The document is signed by Aslam Baig, representing AMB Contractors Inc., and provides the company's address and taxpayer ID. The project is identified by Project ID 22-02603, and the school address is 2703 Webster Avenue, Bronx, NY 10458. Below is a summary of the payroll details:

| Employee | Role | Hours | Rate | Gross Pay | Net Pay | Source |
|----------|------|-------|------|-----------|---------|--------|
| Zenovii Dziuba | Laborer | 16 | $94.43 per hour | $1,510.88 | $1,143.20 | CPRPreviewSCA_6.pdf (Section 0) |
| Ermilo Lopez | Laborer | 8 | $94.43 per hour | $755.44 | $608.65 | CPRPreviewSCA_6.pdf (Section 0) |

<!-- Assumed currency as USD based on context -->"""
    )

    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional personal assistant. Provide clear, concise, and accurate responses based strictly on the provided context and query. "
                        "The context has been preprocessed to minimize OCR errors, but minor issues may remain. "
                        "Structure responses with complete sentences, proper grammar, and correct spelling. "
                        "Remove spaces between characters in words, names, or numbers (e.g., 'D z i u b a' → 'Dziuba', '5 0 . 4 3' → '50.43'). "
                        "Format numbers correctly, using a single '$' for currency (e.g., '$1,988.16', '$94.43 per hour'). "
                        "For payroll-related queries, present details in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay, Source. "
                        "Validate numerical consistency: Gross Pay = Hours × Rate, Net Pay < Gross Pay. Use context-provided rates (e.g., $94.43 per hour) if available. "
                        "Only include total expenditure if explicitly requested in the query (e.g., 'total', 'expenditure', 'spent'). "
                        "Estimate missing hours or rates using context data, noting assumptions. "
                        "Do not speculate or add information beyond the context or query. "
                        "Note limitations due to minor OCR errors if applicable."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=settings.max_completion_tokens,
            temperature=0.3
        )
        response_text = response.choices[0].message.content.strip()
        response_text = clean_response(response_text)  # Apply post-processing
        logger.debug(f"Generated response: {response_text[:500]}...")
        return response_text
    except Exception as e:
        logger.error(f"Failed to generate response: {str(e)}")
        raise

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# API Endpoints
@app.post("/process_file", response_model=Dict[str, str])
async def process_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    api_key: str = Depends(validate_api_key)
):
    """Process uploaded file and extract knowledge"""
    if file.size > settings.max_document_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.max_document_size//(1024*1024)}MB limit"
        )

    # Check for existing file
    existing_file = next(
        (f for f in state_manager.file_metadata
         if f['filename'] == file.filename and f['user_id'] == user_id),
        None
    )
    if existing_file:
        logger.info(f"File {file.filename} already exists for user {user_id}")
        return {
            "status": "success",
            "file_id": existing_file['file_id'],
            "filename": file.filename
        }

    file_ext = os.path.splitext(file.filename)[1].lower()
    file_id = str(uuid.uuid4())
    temp_path = Path(settings.temp_upload_dir) / f"{file_id}{file_ext}"

    try:
        # Save and convert file
        file_content = await file.read()
        temp_path.write_bytes(file_content)

        # Use OCR for PDFs, existing converters for other formats
        if file_ext in settings.supported_extensions['pdfs']:
            logger.info(f"Processing PDF {file.filename} with OCR")
            markdown_content = ocr_processor.process_pdf(str(temp_path))
        else:
            converter = get_file_converter(file_ext)
            if not converter:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_ext}"
                )
            markdown_content = converter(str(temp_path))

        # Preprocess OCR text for images and PDFs
        is_ocr_likely = file_ext in settings.supported_extensions['images'] or file_ext in settings.supported_extensions['pdfs']
        if is_ocr_likely:
            cleaned_markdown = await preprocess_ocr_text(markdown_content)
        else:
            cleaned_markdown = text_processor.clean_markdown(markdown_content)

        # Generate embeddings for chunks
        chunk_embeddings = await text_processor.generate_embeddings(cleaned_markdown)
        if not chunk_embeddings:
            raise ValueError("No embeddings generated for the document")

        # Store each chunk in Qdrant
        for i, (chunk_text, embedding) in enumerate(chunk_embeddings):
            # Generate a valid UUID for the chunk
            chunk_id = str(uuid.uuid5(uuid.UUID(file_id), f"chunk_{i}"))
            
            # Extract entities and relationships for this chunk
            entities = await text_processor.extract_entities(chunk_text)
            relationships = await text_processor.extract_relationships({"content": chunk_text, "chunk_index": i})

            # Store in Qdrant
            qdrant_handler.store_chunk(
                document_id=file_id,
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                embedding=embedding,
                metadata={
                    "filename": file.filename,
                    "user_id": user_id,
                    "chunk_index": i,
                    "parent_section": text_processor._extract_section(chunk_text),
                    "entities": entities,
                    "relationships": relationships
                }
            )

            # Index entities and relationships in knowledge graph
            for entity in entities:
                knowledge_graph.add_relationship(
                    source=entity,
                    target=file_id,
                    relationship="appears_in"
                )

            for rel in relationships:
                knowledge_graph.add_relationship(
                    source=rel['subject'],
                    target=rel['object'],
                    relationship=rel['predicate']
                )

        # Store metadata
        state_manager.file_metadata.append({
            "file_id": file_id,
            "filename": file.filename,
            "file_type": file_ext,
            "upload_date": datetime.datetime.now().isoformat(),
            "content": base64.b64encode(file_content).decode(),
            "markdown_content": cleaned_markdown,
            "user_id": user_id,
            "size": file.size,
            "checksum": hashlib.md5(file_content).hexdigest(),
            "chunks": len(chunk_embeddings)
        })

        state_manager.save()
        logger.info(f"Processed file: {file.filename} (ID: {file_id}) with {len(chunk_embeddings)} chunks")
        return {
            "status": "success",
            "file_id": file_id,
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"File processing failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.error(f"Temp file cleanup failed for {temp_path}: {str(e)}")

@app.post("/search", response_model=Dict[str, Any])
async def search_documents(
    query: str = Form(...),
    user_id: str = Form(...),
    file_ids: Optional[List[str]] = Form(None),
    limit: int = Form(5),
    use_graph: bool = Form(True),
    api_key: str = Depends(validate_api_key)
):
    """Search documents with dual-level retrieval"""
    try:
        # Clean query before processing
        cleaned_query = clean_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Cleaned query for search: {cleaned_query}")
        
        # Entity-based retrieval from knowledge graph
        entity_results = []
        if use_graph:
            entities = await text_processor.extract_entities(cleaned_query)
            for entity in entities:
                related_entities = knowledge_graph.find_related_entities(entity)
                if related_entities:
                    entity_results.extend(
                        await qdrant_handler.search_entities(
                            entities=related_entities,
                            user_id=user_id,
                            file_ids=file_ids,
                            limit=limit
                        )
                    )

        # Vector-based retrieval
        query_chunks = split_large_text(cleaned_query)
        vector_results = []

        for chunk in query_chunks:
            embedding = generate_embeddings_batch([chunk])[0]
            query_filter = {
                "must": [
                    {"key": "user_id", "match": {"value": user_id}},
                ]
            }
            if file_ids:
                query_filter["must"].append(
                    {"key": "document_id", "match": {"any": file_ids}}
                )

            results = qdrant_handler.client.search(
                collection_name=settings.qdrant_collection,
                query_vector=embedding,
                query_filter=query_filter,
                limit=limit
            )
            vector_results.extend(results)

        # Combine and rank results
        combined_results = rank_results(entity_results, vector_results, limit)

        return {
            "status": "success",
            "results": format_search_results(combined_results)
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Dict[str, Any])
async def chat_with_documents(
    query: str = Form(...),
    user_id: str = Form(...),
    file_ids: Optional[List[str]] = Form(None),
    chat_id: Optional[str] = Form(None),
    api_key: str = Depends(validate_api_key)
):
    """Chat with document context"""
    try:
        # Clean query before processing
        cleaned_query = clean_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Cleaned query for chat: {cleaned_query}")

        # Perform dual-level retrieval
        search_results = await search_documents(
            query=cleaned_query,
            user_id=user_id,
            file_ids=file_ids,
            limit=10,
            use_graph=True
        )

        if not search_results["results"]:
            return {
                "response": "No relevant information found in documents.",
                "chat_id": chat_id or str(uuid.uuid4()),
                "sources": []
            }

        # Build context with chunked approach
        context = await build_chat_context(search_results["results"])

        # Generate response
        response = generate_coherent_response(cleaned_query, context)

        # Update chat history
        chat_id = chat_id or str(uuid.uuid4())
        if chat_id not in state_manager.chat_sessions:
            state_manager.chat_sessions[chat_id] = {
                "chat_id": chat_id,
                "user_id": user_id,
                "messages": [],
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "document_ids": file_ids or []
            }

        state_manager.chat_sessions[chat_id]["messages"].append({
            "role": "user",
            "content": query,  # Store original query for history
            "timestamp": datetime.datetime.now().isoformat()
        })
        state_manager.chat_sessions[chat_id]["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        state_manager.chat_sessions[chat_id]["updated_at"] = datetime.datetime.now().isoformat()

        state_manager.save()
        return {
            "response": response,
            "chat_id": chat_id,
            "sources": search_results["results"]
        }

    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=Dict[str, Any])
async def list_documents(
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    """List uploaded documents for a user"""
    try:
        user_docs = [
            f for f in state_manager.file_metadata
            if f.get("user_id") == user_id
        ]
        return {
            "status": "success",
            "documents": [
                {
                    "file_id": f["file_id"],
                    "filename": f["filename"],
                    "file_type": f["file_type"],
                    "upload_date": f["upload_date"],
                    "size": f["size"]
                }
                for f in user_docs
            ]
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )

@app.delete("/documents/{file_id}", response_model=Dict[str, str])
async def delete_document(
    file_id: str,
    user_id: str,
    api_key: str = Depends(validate_api_key)
):
    """Delete a document and its chunks"""
    try:
        await qdrant_handler.delete_by_document_id(file_id)
        state_manager.file_metadata = [
            f for f in state_manager.file_metadata
            if f["file_id"] != file_id
        ]
        state_manager.save()
        logger.info(f"Deleted document: {file_id}")
        return {"status": "success", "file_id": file_id}
    except Exception as e:
        logger.error(f"Failed to delete document {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@app.get("/preview/{file_id}")
async def preview_file(file_id: str, user_id: str, api_key: str = Depends(validate_api_key)):
    """Stream file content for preview"""
    file_meta = next((f for f in state_manager.file_metadata if f['file_id'] == file_id and f['user_id'] == user_id), None)
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    mime_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".heic": "image/heic",
        ".webp": "image/webp",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".odt": "application/vnd.oasis.opendocument.text",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".ods": "application/vnd.oasis.opendocument.spreadsheet",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".rtf": "application/rtf"
    }
    return StreamingResponse(BytesIO(base64.b64decode(file_meta['content'])), media_type=mime_map.get(file_meta['file_type'], "application/octet-stream"))

@app.get("/knowledge_graph", response_model=Dict[str, Any])
async def get_knowledge_graph(
    user_id: str,
    file_id: Optional[str] = None,
    api_key: str = Depends(validate_api_key)
):
    """Retrieve knowledge graph data for visualization"""
    try:
        nodes = []
        edges = []
        file_map = {f['file_id']: f['filename'] for f in state_manager.file_metadata}

        # Filter nodes and edges by user_id and optionally file_id
        for node, data in knowledge_graph.graph.nodes(data=True):
            node_type = data.get('type', 'entity')
            if node_type == 'entity' and (not file_id or any(
                edge[1] == file_id for edge in knowledge_graph.graph.edges(node)
            )):
                nodes.append({
                    "id": node,
                    "label": node,
                    "type": node_type
                })

        # Include document nodes if they match file_id or user_id
        if file_id:
            if file_id in file_map:
                nodes.append({
                    "id": file_id,
                    "label": file_map[file_id],
                    "type": "entity"
                })
        else:
            for file in state_manager.file_metadata:
                if file['user_id'] == user_id:
                    nodes.append({
                        "id": file['file_id'],
                        "label": file['filename'],
                        "type": "entity"
                    })

        # Collect edges
        for source, target, data in knowledge_graph.graph.edges(data=True):
            if file_id and target != file_id and source != file_id:
                continue
            if any(f['file_id'] == target and f['user_id'] == user_id for f in state_manager.file_metadata) or \
               any(f['file_id'] == source and f['user_id'] == user_id for f in state_manager.file_metadata):
                edges.append({
                    "from": source,
                    "to": target,
                    "label": data.get("relationship", "related_to"),
                    "weight": data.get("weight", 1.0)
                })

        logger.info(f"Retrieved knowledge graph for user_id={user_id}, file_id={file_id}")
        return {
            "status": "success",
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        logger.error(f"Failed to retrieve knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)