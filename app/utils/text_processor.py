import re
import logging
import spacy
import tiktoken
from typing import List, Tuple, Dict, Any
from openai import AsyncOpenAI, OpenAIError
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, openai_api_key: str):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Spacy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Spacy model: {str(e)}")
            raise

        try:
            self.client = AsyncOpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise

        try:
            self.tokenizer = tiktoken.encoding_for_model(settings.openai_embedding_model)
            logger.info(f"Tokenizer initialized for OpenAI model: {settings.openai_embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {str(e)}")
            raise

        self.max_tokens = settings.max_embedding_tokens // 2  # 4095 for text-embedding-3-small
        self.overlap = 200

    def chunk_text(self, text: str, max_tokens: int = None) -> List[Dict[str, Any]]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return [{"content": text, "start": 0, "end": len(text)}]

            chunks = []
            current_pos = 0
            text_length = len(text)

            while current_pos < text_length:
                end_pos = current_pos
                current_tokens = 0
                doc = self.nlp(text[current_pos:])

                for sent in doc.sents:
                    sent_text = sent.text
                    sent_tokens = len(self.tokenizer.encode(sent_text))
                    if current_tokens + sent_tokens > max_tokens:
                        if current_tokens > 0:
                            chunk_text = text[current_pos:end_pos]
                            chunks.append({"content": chunk_text.strip(), "start": current_pos, "end": end_pos})
                            current_pos = max(end_pos - self.overlap, 0)
                            current_tokens = 0
                            break
                        else:
                            sub_chunks = []
                            sub_start = current_pos
                            for i in range(0, len(sent_text), max_tokens):
                                sub_chunk = sent_text[i:i + max_tokens]
                                sub_end = sub_start + len(sub_chunk)
                                sub_chunks.append({"content": sub_chunk.strip(), "start": sub_start, "end": sub_end})
                                sub_start = sub_end
                            chunks.extend(sub_chunks)
                            current_pos = sub_end
                            current_tokens = 0
                            break
                    else:
                        current_tokens += sent_tokens
                        end_pos += len(sent_text)

                if current_tokens > 0 and end_pos > current_pos:
                    chunk_text = text[current_pos:end_pos]
                    chunks.append({"content": chunk_text.strip(), "start": current_pos, "end": end_pos})
                    current_pos = max(end_pos - self.overlap, 0)
                    current_tokens = 0

                if end_pos >= text_length:
                    break

            logger.info(f"Chunked text into {len(chunks)} chunks with max {max_tokens} tokens")
            return chunks
        except Exception as e:
            logger.error(f"Failed to chunk text: {str(e)}")
            raise

    async def generate_embeddings(self, text: str) -> List[Tuple[str, List[float]]]:
        try:
            chunks = self.chunk_text(text)
            embeddings = []
            for chunk in chunks:
                try:
                    response = await self.client.embeddings.create(
                        input=chunk["content"],
                        model=settings.openai_embedding_model,
                        encoding_format="float",
                        dimensions=1024  # Match Qdrant collection configuration
                    )
                    embedding = response.data[0].embedding
                    logger.info(f"Generated embedding for chunk with {len(self.tokenizer.encode(chunk['content']))} tokens, dimensions: {len(embedding)}")
                    embeddings.append((chunk["content"], embedding))
                except OpenAIError as e:
                    if "maximum context length" in str(e):
                        logger.warning(f"Chunk too large, retrying with smaller size: {str(e)}")
                        sub_chunks = self.chunk_text(chunk["content"], max_tokens=self.max_tokens // 2)
                        for sub_chunk in sub_chunks:
                            response = await self.client.embeddings.create(
                                input=sub_chunk["content"],
                                model=settings.openai_embedding_model,
                                encoding_format="float",
                                dimensions=1024  # Match Qdrant collection configuration
                            )
                            embedding = response.data[0].embedding
                            logger.info(f"Generated embedding for sub-chunk with {len(self.tokenizer.encode(sub_chunk['content']))} tokens, dimensions: {len(embedding)}")
                            embeddings.append((sub_chunk["content"], embedding))
                    else:
                        logger.error(f"Embedding generation failed for chunk: {str(e)}")
                        raise
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    def clean_markdown(self, text: str) -> str:
        try:
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)
            text = re.sub(r'\n{2,}', '\n\n', text)
            logger.info("Cleaned markdown text")
            return text
        except Exception as e:
            logger.error(f"Failed to clean markdown: {str(e)}")
            raise

    async def preprocess_ocr_text(self, text: str, file_id: str, filename: str) -> str:
        try:
            prompt = (
                f"Input Text:\n{text[:settings.max_completion_tokens]}\n\n"
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

            response = await self.client.chat.completions.create(
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
            cleaned_text = response.choices[0].message.content.strip()
            logger.info(f"Preprocessed OCR text for {filename}")
            return cleaned_text
        except Exception as e:
            logger.error(f"Failed to preprocess OCR text for {filename}: {str(e)}")
            raise

    def _extract_section(self, text: str) -> str:
        try:
            doc = self.nlp(text)
            for token in doc:
                if token.text.startswith("#"):
                    return token.text.lstrip("#").strip()
            return "Main Section"
        except Exception as e:
            logger.error(f"Failed to extract section: {str(e)}")
            return "Main Section"

    async def extract_entities(self, text: str) -> List[str]:
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "DATE", "MONEY"]:
                    entities.append(ent.text)
            entities = list(set(entities))  # Remove duplicates
            logger.info(f"Extracted {len(entities)} entities from text")
            return entities
        except Exception as e:
            logger.error(f"Failed to extract entities: {str(e)}")
            return []

    async def extract_relationships(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        try:
            text = data["content"]
            chunk_index = str(data.get("chunk_index", 0))  # Cast to string to match SearchResult schema
            doc = self.nlp(text)
            relationships = []
            for sent in doc.sents:
                entities = [ent.text for ent in sent.ents if ent.label_ in ["PERSON", "ORG"]]
                if len(entities) >= 2:
                    for i in range(len(entities) - 1):
                        relationships.append({
                            "subject": entities[i],
                            "object": entities[i + 1],
                            "predicate": "related_to",
                            "chunk_index": chunk_index
                        })
            logger.info(f"Extracted {len(relationships)} relationships from text")
            return relationships
        except Exception as e:
            logger.error(f"Failed to extract relationships: {str(e)}")
            return []