import re
import tiktoken
import openai
import os
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = "text-embedding-3-small"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set")
            raise Exception("OPENAI_API_KEY is not set in environment variables")
        try:
            openai.api_key = api_key
            # Test API key with a simple request
            openai.Model.list()
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise Exception(f"OpenAI client initialization failed. Please check OPENAI_API_KEY and library compatibility: {str(e)}")

    def clean_markdown(self, markdown_text):
        # Remove excessive whitespace and newlines
        cleaned_text = re.sub(r'\n\s*\n+', '\n\n', markdown_text.strip())
        # Remove markdown artifacts (e.g., extra asterisks, underscores)
        cleaned_text = re.sub(r'(\*|_){2,}', '', cleaned_text)
        return cleaned_text

    def chunk_text(self, text, max_tokens=300):
        tokens = self.encoder.encode(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_section = "None"

        lines = text.split('\n')
        for line in lines:
            if line.startswith('#'):
                if current_chunk and current_tokens > 0:
                    chunks.append({
                        'chunk_id': str(uuid.uuid4()),
                        'document_id': None,  # To be set later
                        'content': '\n'.join(current_chunk),
                        'embedding': None,  # To be set later
                        'parent_section': current_section,
                        'chunk_index': len(chunks)
                    })
                    current_chunk = []
                    current_tokens = 0
                current_section = line.strip()
            else:
                line_tokens = len(self.encoder.encode(line))
                if current_tokens + line_tokens > max_tokens and current_chunk:
                    chunks.append({
                        'chunk_id': str(uuid.uuid4()),
                        'document_id': None,
                        'content': '\n'.join(current_chunk),
                        'embedding': None,
                        'parent_section': current_section,
                        'chunk_index': len(chunks)
                    })
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(line)
                current_tokens += line_tokens
        
        if current_chunk:
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'document_id': None,
                'content': '\n'.join(current_chunk),
                'embedding': None,
                'parent_section': current_section,
                'chunk_index': len(chunks)
            })
        
        return chunks

    def generate_embeddings(self, chunks):
        for chunk in chunks:
            try:
                response = openai.Embedding.create(
                    model=self.embedding_model,
                    input=chunk['content']
                )
                chunk['embedding'] = response['data'][0]['embedding']
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk['chunk_index']}: {str(e)}")
                raise Exception(f"Failed to generate embedding for chunk {chunk['chunk_index']}: {str(e)}")
        return chunks