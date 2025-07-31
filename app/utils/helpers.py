import logging
import re
import datetime
import psycopg2
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
from app.utils.text_processor import TextProcessor
from openai import OpenAI
import tiktoken

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Initialize TextProcessor
text_processor = TextProcessor()

# Configure logging
logger = logging.getLogger(__name__)

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def preprocess_ocr_text(text: str) -> str:
    def split_large_text(text: str, max_tokens: int = settings.max_embedding_tokens // 2) -> list[str]:
        table_pattern = r'(\|.*?\|\n(?:\|[-: ]+\|\n)+.*?)(?=\n\n|\Z)'
        tables = re.findall(table_pattern, text, re.DOTALL)
        non_table_text = re.sub(table_pattern, 'TABLE_PLACEHOLDER', text)

        chunks = []
        current_chunk = ""
        token_count = 0
        placeholder_count = 0

        doc = text_processor.nlp(non_table_text)
        for sent in doc.sents:
            sent_text = sent.text
            if 'TABLE_PLACEHOLDER' in sent_text:
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
                    sub_chunks = [sent_text[i:i + max_tokens] for i in range(0, len(sent_text), max_tokens)]
                    chunks.extend(sub_chunks)
                    continue

            current_chunk += sent_text + "\n"
            token_count += sent_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        while placeholder_count < len(tables):
            table_tokens = len(tokenizer.encode(tables[placeholder_count]))
            if table_tokens <= max_tokens:
                chunks.append(tables[placeholder_count])
            else:
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

    if not settings.openai_enabled:
        logger.error("OpenAI is not enabled. Please set OPENAI_ENABLED to true.")
        raise Exception("Invalid configuration: OPENAI_ENABLED must be set to true.")

    chunks = split_large_text(text, max_tokens=settings.max_embedding_tokens // 2)
    cleaned_chunks = []

    client = OpenAI(api_key=settings.openai_api_key)

    for chunk in chunks:
        try:
            prompt = (
                f"Input Text:\n{chunk}\n\n"
                "You are a document cleaner. Remove all OCR artifacts first, then re-format the remainder.\n\n"
                "Step-0 (MANDATORY): Collapse every space inside a token (words, names, numbers, currency, dates). "
                "Example: ‘1 , 0 0 0 , 0 0 0’ → ‘1,000,000’, ‘A d d i t i o n a l’ → ‘Additional’. "
                "Do not remove spaces that separate distinct tokens.\n\n"
                "Steps 1-11:\n"
                "1. Reconstruct the text into clear, grammatical, logically structured content.\n"
                "2. Standardize numbers and currency (single ‘$’, commas for thousands, two decimals).\n"
                "3. Rejoin broken names/words that remain after Step-0.\n"
                "4. Separate metadata labels (Date, Signature, etc.) from the preceding name/value.\n"
                "5. Preserve markdown headings, lists, and tables; ensure tables are well-aligned.\n"
                "6. If payroll data (employee, role, hours, rate, gross/net pay) is present:\n"
                "   - Render it in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay.\n"
                "   - Validate: Gross Pay ≈ Hours × Rate, Net Pay < Gross Pay. Note any assumptions.\n"
                "7. If not payroll-related, simply return clean markdown prose.\n"
                "8. Remove gibberish or noise; keep all meaningful data.\n"
                "9. Comment assumptions with <!-- ... -->.\n"
                "10. Return only the cleaned markdown.\n"
                "11. Stay under the token budget.\n"
            )

            response = client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at cleaning and reconstructing noisy OCR-extracted text. "
                            "Correct OCR errors, including spaces between characters in words, names, or numbers. "
                            "Format numbers and names properly, ensuring single '$' for currency, and structure the output in clear, coherent markdown. "
                            "Preserve meaningful information and structural elements (e.g., lists, tables). "
                            "For payroll data, format details in a markdown table with columns: Employee, Role, Hours, Rate, Gross Pay, Net Pay. "
                            "Validate numerical consistency: Gross Pay = Hours × Rate, Net Pay < Gross Pay. "
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
        except Exception as e:
            logger.error(f"Failed to preprocess OCR chunk: {str(e)}")
            raise

    return "\n\n".join(cleaned_chunks)