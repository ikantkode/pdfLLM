import re
import json
import logging
from typing import List, Dict, Any
import spacy
import openai
from config import settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Spacy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Spacy model: {str(e)}")
            raise

    def clean_markdown(self, text: str) -> str:
        """Clean markdown text by removing excessive whitespace and special characters"""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = re.sub(r'#+', '#', text)   # Normalize headers
        text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)  # Fix spaced characters (e.g., "D z i u b a" → "Dziuba")
        text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)  # Fix spaced numbers (e.g., "1 , 510 . 88" → "1,510.88")
        text = text.strip()
        return text

    def chunk_text(self, text: str, max_tokens: int = settings.max_embedding_tokens) -> List[Dict]:
        """Split text into chunks, preserving table structures"""
        # Detect markdown tables
        table_pattern = r'(\|.*?\|\n(?:\|[-: ]+\|\n)+.*?)(?=\n\n|\Z)'
        tables = re.findall(table_pattern, text, re.DOTALL)
        non_table_text = re.sub(table_pattern, 'TABLE_PLACEHOLDER', text)

        chunks = []
        current_chunk = ""
        token_count = 0
        chunk_index = 0
        placeholder_count = 0

        # Split non-table text by sentences
        doc = self.nlp(non_table_text)
        for sent in doc.sents:
            sent_text = sent.text
            if 'TABLE_PLACEHOLDER' in sent_text:
                # Replace placeholder with actual table
                if placeholder_count < len(tables):
                    table_text = tables[placeholder_count]
                    table_tokens = len(self.nlp.tokenizer(table_text))
                    if table_tokens > max_tokens:
                        # Split table into rows
                        rows = table_text.split('\n')
                        sub_chunk = ""
                        sub_tokens = 0
                        for row in rows:
                            row_tokens = len(self.nlp.tokenizer(row))
                            if sub_tokens + row_tokens > max_tokens:
                                if sub_chunk:
                                    chunks.append({
                                        "content": sub_chunk.strip(),
                                        "chunk_index": chunk_index,
                                        "parent_section": self._extract_section(sub_chunk),
                                        "entities": [],
                                        "relationships": []
                                    })
                                    chunk_index += 1
                                    sub_chunk = ""
                                    sub_tokens = 0
                                sub_chunk += row + "\n"
                                sub_tokens += row_tokens
                            else:
                                sub_chunk += row + "\n"
                                sub_tokens += row_tokens
                        if sub_chunk:
                            chunks.append({
                                "content": sub_chunk.strip(),
                                "chunk_index": chunk_index,
                                "parent_section": self._extract_section(sub_chunk),
                                "entities": [],
                                "relationships": []
                            })
                            chunk_index += 1
                        placeholder_count += 1
                        continue
                    else:
                        sent_text = sent_text.replace('TABLE_PLACEHOLDER', table_text)
                        placeholder_count += 1

            sent_tokens = len(self.nlp.tokenizer(sent_text))
            if token_count + sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "parent_section": self._extract_section(current_chunk),
                        "entities": [],
                        "relationships": []
                    })
                    chunk_index += 1
                    current_chunk = ""
                    token_count = 0

            current_chunk += sent_text + "\n"
            token_count += sent_tokens

        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "chunk_index": chunk_index,
                "parent_section": self._extract_section(current_chunk),
                "entities": [],
                "relationships": []
            })

        # Add remaining tables
        while placeholder_count < len(tables):
            table_text = tables[placeholder_count]
            table_tokens = len(self.nlp.tokenizer(table_text))
            if table_tokens <= max_tokens:
                chunks.append({
                    "content": table_text.strip(),
                    "chunk_index": chunk_index,
                    "parent_section": self._extract_section(table_text),
                    "entities": [],
                    "relationships": []
                })
                chunk_index += 1
            else:
                # Split table into rows
                rows = table_text.split('\n')
                sub_chunk = ""
                sub_tokens = 0
                for row in rows:
                    row_tokens = len(self.nlp.tokenizer(row))
                    if sub_tokens + row_tokens > max_tokens:
                        if sub_chunk:
                            chunks.append({
                                "content": sub_chunk.strip(),
                                "chunk_index": chunk_index,
                                "parent_section": self._extract_section(sub_chunk),
                                "entities": [],
                                "relationships": []
                            })
                            chunk_index += 1
                            sub_chunk = ""
                            sub_tokens = 0
                        sub_chunk += row + "\n"
                        sub_tokens += row_tokens
                    else:
                        sub_chunk += row + "\n"
                        sub_tokens += row_tokens
                if sub_chunk:
                    chunks.append({
                        "content": sub_chunk.strip(),
                        "chunk_index": chunk_index,
                        "parent_section": self._extract_section(sub_chunk),
                        "entities": [],
                        "relationships": []
                    })
                    chunk_index += 1
            placeholder_count += 1

        logger.debug(f"Chunked text into {len(chunks)} chunks")
        return chunks

    def _extract_section(self, text: str) -> str:
        """Extract section identifier from text"""
        lines = text.split('\n')
        for line in lines:
            if line.startswith('#'):
                return line.strip('# ').strip()
        return "N/A"

    async def extract_entities(self, text: str) -> List[str]:
        """Extract entities using Spacy"""
        try:
            doc = self.nlp(text)
            entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "PRODUCT", "EVENT"]]
            logger.debug(f"Extracted entities: {entities}")
            return list(set(entities))  # Remove duplicates
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def extract_relationships(self, chunk: Dict) -> List[Dict]:
        """Extract relationships using OpenAI API"""
        prompt = (
            "Extract precise relationships (e.g., 'person works at organization', 'entity is located in place', 'organization owns product') "
            "from the following text. Return a JSON object with a 'relationships' key containing a list of "
            "{subject, predicate, object} dictionaries. Ensure the response is valid JSON and relationships are specific and meaningful. "
            "If no relationships are found, return an empty list under 'relationships'. "
            "Do not include markdown code blocks or extra text.\n\n"
            f"Text: {chunk['content']}\n\n"
            "Example output:\n"
            "{\"relationships\": [{\"subject\": \"John Doe\", \"predicate\": \"works at\", \"object\": \"Acme Corp\"}, {\"subject\": \"Acme Corp\", \"predicate\": \"owns\", \"object\": \"Product X\"}]}"
        )

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in relationship extraction. Return valid JSON with a 'relationships' key "
                            "containing a list of dictionaries with 'subject', 'predicate', and 'object' keys. "
                            "If no relationships are found, return {\"relationships\": []}. "
                            "Do not include markdown code blocks or extra text."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )

            raw_response = response.choices[0].message.content.strip()

            if not raw_response:
                logger.error(f"Empty response for chunk {chunk['chunk_index']}")
                return []

            try:
                result = json.loads(raw_response)
                if not isinstance(result, dict) or 'relationships' not in result:
                    logger.error(f"Invalid JSON structure for chunk {chunk['chunk_index']}: missing 'relationships' key")
                    logger.debug(f"Problematic response: {raw_response}")
                    return []

                if not isinstance(result['relationships'], list):
                    logger.error(f"'relationships' is not a list in chunk {chunk['chunk_index']}")
                    logger.debug(f"Problematic response: {raw_response}")
                    return []

                valid_relationships = []
                for rel in result['relationships']:
                    if isinstance(rel, dict) and all(key in rel for key in ['subject', 'predicate', 'object']):
                        valid_relationships.append({
                            'subject': rel['subject'],
                            'predicate': rel['predicate'],
                            'object': rel['object']
                        })
                    else:
                        logger.warning(f"Invalid relationship format in chunk {chunk['chunk_index']}: {rel}")

                logger.debug(f"Extracted relationships for chunk {chunk['chunk_index']}: {valid_relationships}")
                return valid_relationships
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for chunk {chunk['chunk_index']}: {str(e)}")
                logger.debug(f"Problematic response: {raw_response}")
                return []
        except Exception as e:
            logger.error(f"Failed to extract relationships for chunk {chunk['chunk_index']}: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """Generate embeddings for chunks and extract entities/relationships"""
        texts = [chunk['content'] for chunk in chunks]
        try:
            response = openai.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = response.data[i].embedding
                chunk['entities'] = await self.extract_entities(chunk['content'])
                chunk['relationships'] = await self.extract_relationships(chunk)
            return chunks
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise