import logging
import base64
import io
import re
from pathlib import Path
from typing import List, Optional
from PIL import Image
import PyPDF2
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor with pytesseract."""
        logger.info("OCRProcessor initialized with pytesseract")

    def process_pdf(self, pdf_path: str) -> str:
        """Process a PDF file and extract text using pytesseract."""
        logger.info(f"Processing PDF {pdf_path} with pytesseract")
        try:
            return self._process_pdf_with_pytesseract(pdf_path)
        except Exception as e:
            logger.error(f"OCR processing failed for {pdf_path}: {str(e)}")
            raise

    def _process_pdf_with_pytesseract(self, pdf_path: str) -> str:
        """Process PDF with pytesseract, converting pages to images first."""
        try:
            images = convert_from_path(pdf_path)
            markdown_content = []

            for page_num, image in enumerate(images, start=1):
                text = pytesseract.image_to_string(image, lang='eng')

                # NEW: Fix OCR artifacts in raw text before further processing
                text = re.sub(r'(\w)\s+(\w)\s+(\w)\s+(\w)', r'\1\2\3\4', text)  # 4-char words
                text = re.sub(r'(\w)\s+(\w)\s+(\w)', r'\1\2\3', text)  # 3-char words
                text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)  # 2-char words
                text = re.sub(r'(\d)\s+([,.])\s+(\d)', r'\1\2\3', text)  # Numbers

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