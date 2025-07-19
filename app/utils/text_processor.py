import re
import json
import logging
from typing import List, Dict, Any, Tuple
import spacy
from openai import OpenAI
import tiktoken
import requests
from config import settings
from tenacity import retry, stop_after_attempt, wait_exponential
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Spacy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Spacy model: {str(e)}")
            raise

        # Validate provider settings
        if settings.openai_enabled and settings.ollama_enabled:
            logger.error("Both OpenAI and Ollama are enabled. Only one provider can be active.")
            raise ValueError("Invalid configuration: Both OPENAI_ENABLED and OLLAMA_ENABLED are set to true.")
        if not settings.openai_enabled and not settings.ollama_enabled:
            logger.error("No provider enabled. Either OPENAI_ENABLED or OLLAMA_ENABLED must be set to true.")
            raise ValueError("Invalid configuration: Neither OPENAI_ENABLED nor OLLAMA_ENABLED is set to true.")

        # Initialize OpenAI client only if OpenAI is enabled
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_enabled else None

        # Select tokenizer based on enabled provider
        try:
            if settings.openai_enabled:
                self.tokenizer = tiktoken.encoding_for_model(settings.openai_embedding_model)
                logger.info(f"Tokenizer initialized for OpenAI model: {settings.openai_embedding_model}")
            else:  # Ollama enabled
                # Ollama uses cl100k_base tokenizer for compatibility
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.info("Tokenizer initialized for Ollama with cl100k_base")
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {str(e)}")
            raise

        self.max_tokens = settings.max_embedding_tokens

    def clean_markdown(self, text: str) -> str:
        """More aggressive cleaning for RAG contexts"""
        # Fix space-separated characters
        text = re.sub(r'(\w)\s+(\w)\s+(\w)\s+(\w)', r'\1\2\3\4', text)
        text = re.sub(r'(\w)\s+(\w)\s+(\w)', r'\1\2\3', text) 
        text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
        
        # Fix numbers
        text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)
        text = re.sub(r'(\d)\s+(\d{3})\s+(\d{3})', r'\1,\2,\3', text)
        
        # Remove OCR artifacts
        text = re.sub(r'[\x00-\x1F]', ' ', text)  # Control chars
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def chunk_text(self, text: str, max_tokens: int = None) -> List[Dict]:
        """Split text into chunks, preserving table structures, using tiktoken for accurate token counting"""
        max_tokens = max_tokens or self.max_tokens
        table_pattern = r'(\|.*?\|\n(?:\|[-: ]+\|\n)+.*?)(?=\n\n|\Z)'
        tables = re.findall(table_pattern, text, re.DOTALL)
        non_table_text = re.sub(table_pattern, 'TABLE_PLACEHOLDER', text)

        chunks = []
        current_chunk = []
        current_token_count = 0
        chunk_index = 0
        placeholder_count = 0

        doc = self.nlp(non_table_text)
        for sent in doc.sents:
            sent_text = sent.text
            if 'TABLE_PLACEHOLDER' in sent_text:
                if placeholder_count < len(tables):
                    table_text = tables[placeholder_count]
                    table_tokens = len(self.tokenizer.encode(table_text))
                    if table_tokens > max_tokens:
                        rows = table_text.split('\n')
                        sub_chunk = []
                        sub_tokens = 0
                        for row in rows:
                            row_tokens = len(self.tokenizer.encode(row))
                            if sub_tokens + row_tokens > max_tokens:
                                if sub_chunk:
                                    chunks.append({
                                        "content": "\n".join(sub_chunk).strip(),
                                        "chunk_index": chunk_index,
                                        "parent_section": self._extract_section("\n".join(sub_chunk)),
                                        "entities": [],
                                        "relationships": []
                                    })
                                    chunk_index += 1
                                    sub_chunk = []
                                    sub_tokens = 0
                                sub_chunk.append(row)
                                sub_tokens += row_tokens
                            else:
                                sub_chunk.append(row)
                                sub_tokens += row_tokens
                        if sub_chunk:
                            chunks.append({
                                "content": "\n".join(sub_chunk).strip(),
                                "chunk_index": chunk_index,
                                "parent_section": self._extract_section("\n".join(sub_chunk)),
                                "entities": [],
                                "relationships": []
                            })
                            chunk_index += 1
                        placeholder_count += 1
                        continue
                    else:
                        sent_text = sent_text.replace('TABLE_PLACEHOLDER', table_text)
                        placeholder_count += 1

            sent_tokens = len(self.tokenizer.encode(sent_text))
            if current_token_count + sent_tokens > max_tokens:
                if current_chunk:
                    chunks.append({
                        "content": "\n".join(current_chunk).strip(),
                        "chunk_index": chunk_index,
                        "parent_section": self._extract_section("\n".join(current_chunk)),
                        "entities": [],
                        "relationships": []
                    })
                    chunk_index += 1
                    current_chunk = []
                    current_token_count = 0

            current_chunk.append(sent_text)
            current_token_count += sent_tokens

        if current_chunk:
            chunks.append({
                "content": "\n".join(current_chunk).strip(),
                "chunk_index": chunk_index,
                "parent_section": self._extract_section("\n".join(current_chunk)),
                "entities": [],
                "relationships": []
            })
            chunk_index += 1

        while placeholder_count < len(tables):
            table_text = tables[placeholder_count]
            table_tokens = len(self.tokenizer.encode(table_text))
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
                rows = table_text.split('\n')
                sub_chunk = []
                sub_tokens = 0
                for row in rows:
                    row_tokens = len(self.tokenizer.encode(row))
                    if sub_tokens + row_tokens > max_tokens:
                        if sub_chunk:
                            chunks.append({
                                "content": "\n".join(sub_chunk).strip(),
                                "chunk_index": chunk_index,
                                "parent_section": self._extract_section("\n".join(sub_chunk)),
                                "entities": [],
                                "relationships": []
                            })
                            chunk_index += 1
                            sub_chunk = []
                            sub_tokens = 0
                        sub_chunk.append(row)
                        sub_tokens += row_tokens
                    else:
                        sub_chunk.append(row)
                        sub_tokens += row_tokens
                if sub_chunk:
                    chunks.append({
                        "content": "\n".join(sub_chunk).strip(),
                        "chunk_index": chunk_index,
                        "parent_section": self._extract_section("\n".join(sub_chunk)),
                        "entities": [],
                        "relationships": []
                    })
                    chunk_index += 1
                placeholder_count += 1

        logger.info(f"Chunked text into {len(chunks)} chunks with max {max_tokens} tokens")
        return chunks

    def _extract_section(self, text: str) -> str:
        """Extract the parent section from markdown text"""
        lines = text.split('\n')
        for line in lines:
            if line.startswith('#'):
                return line.strip('# ').strip() or "N/A"
        return "N/A"

    async def extract_entities(self, text: str) -> List[str]:
        """Extract and sanitize entities from text, removing newlines and metadata like 'Date'"""
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "NORP", "PRODUCT", "EVENT"]:
                    cleaned_entity = self._sanitize_entity(ent.text)
                    if (cleaned_entity and len(cleaned_entity) <= 255 and 
                        not self._contains_metadata(cleaned_entity)):
                        entities.append(cleaned_entity)
            entities = list(set(entities))
            logger.debug(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def extract_relationships(self, chunk: Dict) -> List[Dict]:
        """Extract relationships from chunk, ensuring subjects and objects are sanitized.
        
        Args:
            chunk: Dictionary containing text content with keys 'content' and 'chunk_index'
            
        Returns:
            List of dictionaries with 'subject', 'predicate', 'object' keys, or empty list on error
        """
        # Validate input
        if not isinstance(chunk, dict) or 'content' not in chunk:
            logger.error("Invalid chunk format - missing 'content' key")
            return []
        
        content = str(chunk['content'])[:10000]  # Truncate very long content to prevent API issues
        chunk_index = chunk.get('chunk_index', 'unknown')
        
        prompt = (
            "Extract precise relationships (e.g., 'person works at organization', 'entity is located in place') "
            "from the following text. Return a JSON object with a 'relationships' key containing a list of "
            "{subject, predicate, object} dictionaries. Follow these rules:\n"
            "1. Ensure response is valid JSON\n"
            "2. Relationships must be specific and meaningful\n"
            "3. Sanitize subjects/objects (remove newlines, tabs, excessive spaces)\n"
            "4. Exclude entities containing 'Date' or 'Signature'\n"
            "5. If no relationships, return {'relationships': []}\n"
            "6. No markdown code blocks or extra text\n\n"
            f"Text: {content}\n\n"
            "Example output:\n"
            "{\"relationships\": [{\"subject\": \"John Doe\", \"predicate\": \"works at\", \"object\": \"Acme Corp\"}]}"
        )

        try:
            if settings.openai_enabled:
                response = self.client.chat.completions.create(
                    model=settings.openai_chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert relationship extractor. Follow these rules:\n"
                                "1. Return ONLY valid JSON with {'relationships': [...]}\n"
                                "2. Each relationship must have subject, predicate, object\n"
                                "3. Sanitize text (no newlines/tabs, minimal spaces)\n"
                                "4. Exclude Date/Signature entities\n"
                                "5. No explanations or extra text\n"
                                "6. If uncertain, omit the relationship"
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=300,
                    temperature=0.2  # Lower temperature for more consistent results
                )
                raw_response = response.choices[0].message.content.strip()
            else:  # Ollama enabled
                response = requests.post(
                    f"http://{settings.ollama_host}:{settings.ollama_port}/api/chat",
                    json={
                        "model": settings.ollama_chat_model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert relationship extractor. Rules:\n"
                                    "1. Return ONLY valid JSON with {'relationships': [...]}\n"
                                    "2. Each relationship must have subject, predicate, object\n"
                                    "3. Sanitize text (no newlines/tabs, minimal spaces)\n"
                                    "4. Exclude Date/Signature entities\n"
                                    "5. No markdown, explanations, or extra text\n"
                                    "6. If uncertain, omit the relationship"
                                )
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "options": {
                            "temperature": 0.2,
                            "num_ctx": 2048
                        },
                        "format": "json",
                        "stream": False
                    },
                    timeout=30  # Add timeout to prevent hanging
                )
                response.raise_for_status()
                raw_response = response.json().get("message", {}).get("content", "").strip()
                logger.debug(f"Raw Ollama response for chunk {chunk_index}: {raw_response[:200]}...")  # Log truncated response

            # Clean and parse response
            cleaned_response = raw_response
            
            # Remove all markdown code blocks
            cleaned_response = re.sub(r'```(json)?\s*|\s*```', '', cleaned_response, flags=re.MULTILINE)
            
            # Remove any non-JSON text before/after the JSON object
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON object found in chunk {chunk_index} response")
                return []
                
            cleaned_response = json_match.group(0)
            
            # Fix common JSON issues
            cleaned_response = re.sub(r',\s*([\]\}])', r'\1', cleaned_response)  # Trailing commas
            cleaned_response = re.sub(r'[\x00-\x1f]', '', cleaned_response)  # Remove control chars

            try:
                result = json.loads(cleaned_response)
                if not isinstance(result, dict):
                    logger.warning(f"Response is not a dictionary in chunk {chunk_index}")
                    return []
                    
                relationships = result.get('relationships', [])
                if not isinstance(relationships, list):
                    logger.warning(f"'relationships' is not a list in chunk {chunk_index}")
                    return []

                valid_relationships = []
                for rel in relationships:
                    if not isinstance(rel, dict):
                        continue
                        
                    try:
                        subject = self._sanitize_entity(rel.get('subject', ''))
                        predicate = str(rel.get('predicate', '')).strip()
                        object_ = self._sanitize_entity(rel.get('object', ''))
                        
                        if (subject and predicate and object_ and
                            len(subject) <= 255 and len(object_) <= 255 and
                            not self._contains_metadata(subject) and
                            not self._contains_metadata(object_)):
                            valid_relationships.append({
                                'subject': subject,
                                'predicate': predicate,
                                'object': object_
                            })
                    except Exception as e:
                        logger.debug(f"Error processing relationship in chunk {chunk_index}: {str(e)}")

                logger.info(f"Extracted {len(valid_relationships)} valid relationships from chunk {chunk_index}")
                return valid_relationships
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode failed for chunk {chunk_index}: {str(e)}")
                logger.debug(f"Problematic response: {cleaned_response[:200]}...")
                return []
                
        except requests.RequestException as e:
            logger.error(f"API request failed for chunk {chunk_index}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing chunk {chunk_index}: {str(e)}", exc_info=True)
            return []

    def _sanitize_entity(self, text: str) -> str:
        """Sanitize entity text by removing unwanted characters and normalizing whitespace"""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[\n\r\t]', ' ', text)  # Remove line breaks and tabs
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()[:255]  # Enforce max length

    def _contains_metadata(self, text: str) -> bool:
        """Check if text contains metadata keywords to exclude"""
        return bool(re.search(r'\b(Date|Signature|Page|Time)\b', text, re.IGNORECASE))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def generate_embeddings(self, text: str) -> List[Tuple[str, List[float]]]:
        """Generate embeddings for text chunks, including entity and relationship extraction"""
        chunks = self.chunk_text(text)
        texts = [chunk['content'] for chunk in chunks]
        try:
            if settings.openai_enabled:
                response = self.client.embeddings.create(
                    input=texts,
                    model=settings.openai_embedding_model,
                    dimensions=1024  # Truncate to 1024 dimensions
                )
                embeddings = [item.embedding for item in response.data]
                logger.info(f"OpenAI embeddings generated with {len(embeddings[0])} dimensions")
            else:  # Ollama enabled
                embeddings = []
                for text in texts:
                    response = requests.post(
                        f"http://{settings.ollama_host}:{settings.ollama_port}/api/embeddings",
                        json={
                            "model": settings.ollama_embedding_model,
                            "prompt": text
                        },
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    logger.debug(f"Ollama response for chunk {chunks[len(embeddings)]['chunk_index']}: {data}")
                    embeddings.append(data["embedding"])
                logger.info(f"Ollama embeddings generated with {len(embeddings[0])} dimensions")

            result = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk['entities'] = await self.extract_entities(chunk['content'])
                chunk['relationships'] = await self.extract_relationships(chunk)
                result.append((chunk['content'], embedding))
                logger.info(f"Generated embedding for chunk {chunk['chunk_index']} with {len(self.tokenizer.encode(chunk['content']))} tokens")
            return result
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
        return embeddings