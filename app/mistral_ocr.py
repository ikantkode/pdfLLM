import logging
import base64
import io
from pathlib import Path
from typing import List, Optional
from PIL import Image
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from mistralai import Mistral

logger = logging.getLogger(__name__)

class MistralOCRProcessor:
    def __init__(self, api_key: str, use_mistral: bool = True):
        """Initialize the OCR processor with Mistral or pytesseract."""
        self.api_key = api_key
        self.use_mistral = use_mistral
        if self.use_mistral:
            try:
                self.client = Mistral(api_key=api_key)
                logger.info("Mistral client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Mistral client: {str(e)}. Falling back to pytesseract.")
                self.use_mistral = False

    def process_pdf(self, pdf_path: str) -> str:
        """Process a PDF file and extract text using Mistral OCR or pytesseract."""
        logger.info(f"Processing PDF {pdf_path} with OCR (Mistral: {self.use_mistral})")
        try:
            if self.use_mistral:
                return self._process_pdf_with_mistral(pdf_path)
            else:
                return self._process_pdf_with_pytesseract(pdf_path)
        except Exception as e:
            logger.error(f"OCR processing failed for {pdf_path}: {str(e)}")
            raise

    def _process_pdf_with_mistral(self, pdf_path: str) -> str:
        """Process PDF with Mistral OCR, batching pages to optimize API calls."""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            markdown_content = []
            batch_size = 5  # Process 5 pages per API call to optimize
            model = "mistral-ocr-latest"

            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                logger.info(f"Processing pages {i + 1} to {min(i + batch_size, len(images))} with Mistral {model}")
                batch_markdown = []

                for page_num, image in enumerate(batch, start=i + 1):
                    # Convert image to base64
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    try:
                        response = self.client.chat.complete(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Extract all text from this image and format it as markdown. "
                                                "Preserve tables, headings, and other structural elements. "
                                                "For tables, use markdown table syntax. "
                                                "If text appears to be payroll-related, identify and format details such as "
                                                "Employee, Role, Hours, Rate, Gross Pay, and Net Pay in a markdown table."
                                            )
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": f"data:image/png;base64,{img_str}"
                                        }
                                    ]
                                }
                            ],
                            max_tokens=4000,
                            temperature=0.3
                        )
                        page_markdown = response.choices[0].message.content.strip()
                        batch_markdown.append(f"## Page {page_num}\n{page_markdown}")
                    except Exception as e:
                        logger.warning(f"Mistral model {model} failed for page {page_num}: {str(e)}. Trying pixtral-12b.")
                        response = self.client.chat.complete(
                            model="pixtral-12b",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Extract all text from this image and format it as markdown. "
                                                "Preserve tables, headings, and other structural elements. "
                                                "For tables, use markdown table syntax. "
                                                "If text appears to be payroll-related, identify and format details such as "
                                                "Employee, Role, Hours, Rate, Gross Pay, and Net Pay in a markdown table."
                                            )
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": f"data:image/png;base64,{img_str}"
                                        }
                                    ]
                                }
                            ],
                            max_tokens=4000,
                            temperature=0.3
                        )
                        page_markdown = response.choices[0].message.content.strip()
                        batch_markdown.append(f"## Page {page_num}\n{page_markdown}")

                markdown_content.extend(batch_markdown)

            result = "\n\n".join(markdown_content)
            logger.info(f"Successfully extracted markdown from PDF using Mistral: {result[:500]}...")
            return result

        except Exception as e:
            logger.error(f"Mistral OCR failed for {pdf_path}: {str(e)}. Falling back to pytesseract.")
            self.use_mistral = False
            return self._process_pdf_with_pytesseract(pdf_path)

    def _process_pdf_with_pytesseract(self, pdf_path: str) -> str:
        """Process PDF with pytesseract, converting pages to images first."""
        try:
            images = convert_from_path(pdf_path)
            markdown_content = []

            for page_num, image in enumerate(images, start=1):
                text = pytesseract.image_to_string(image, lang='eng')
                # Basic markdown formatting for pytesseract output
                lines = text.split('\n')
                formatted_lines = []
                in_table = False
                table_rows = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        if in_table and table_rows:
                            formatted_lines.append(self._format_table(table_rows))
                            table_rows = []
                            in_table = False
                        formatted_lines.append("")
                        continue
                    if line.startswith("|") and line.endswith("|"):
                        in_table = True
                        table_rows.append(line)
                    else:
                        if in_table and table_rows:
                            formatted_lines.append(self._format_table(table_rows))
                            table_rows = []
                            in_table = False
                        formatted_lines.append(line)

                if table_rows:
                    formatted_lines.append(self._format_table(table_rows))

                page_markdown = "\n".join(formatted_lines).strip()
                markdown_content.append(f"## Page {page_num}\n{page_markdown}")

            result = "\n\n".join(markdown_content)
            logger.info(f"Successfully extracted markdown from PDF using pytesseract: {result[:500]}...")
            return result

        except Exception as e:
            logger.error(f"pytesseract OCR failed for {pdf_path}: {str(e)}")
            raise

    def _format_table(self, rows: List[str]) -> str:
        """Format detected table rows into markdown table syntax."""
        if not rows:
            return ""
        try:
            # Assume first row is header
            headers = rows[0].strip("|").split("|")
            headers = [h.strip() for h in headers]
            table = ["| " + " | ".join(headers) + " |"]
            table.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows[1:]:
                cells = row.strip("|").split("|")
                cells = [c.strip() for c in cells]
                if len(cells) == len(headers):
                    table.append("| " + " | ".join(cells) + " |")
            return "\n".join(table)
        except Exception as e:
            logger.warning(f"Failed to format table: {str(e)}")
            return "\n".join(rows)