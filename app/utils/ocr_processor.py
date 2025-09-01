import logging
import subprocess
import os
import tempfile
import img2pdf
import pymupdf4llm
import pymupdf
import concurrent.futures
import time
import re
from typing import Tuple, Optional, List
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor with ocrmypdf and related tools."""
        self.executor = ThreadPoolExecutor(max_workers=10)
        logger.info("OCRProcessor initialized with ocrmypdf support")

    def has_embedded_text(self, pdf_path: str) -> bool:
        """Check if PDF has embedded text."""
        try:
            doc = pymupdf.open(pdf_path)
            for page in doc:
                text = page.get_text("text").strip()
                if text:
                    doc.close()
                    return True
            doc.close()
            return False
        except Exception as e:
            logger.error(f"Error checking embedded text in {pdf_path}: {e}")
            return False

    def clean_markdown(self, md_text: str) -> str:
        """Enhanced markdown cleaning that preserves tables."""
        if not md_text:
            return ""
        lines = md_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Fix table formatting
            if '|' in line and not line.startswith('#'):
                line = re.sub(r'\s*\|\s*', '|', line)
                line = re.sub(r'\|{2,}', '|', line)
                line = re.sub(r'^\||\|$', '', line)
                line = '| ' + line + ' |'
                # Fix table separators
                if re.match(r'^\|[-:\s]+\|', line):
                    line = re.sub(r'[-:\s]+', '-', line)
            # Clean up but preserve table separators
            if not (line.startswith('|') and '---' in line):
                line = re.sub(r'-{4,}', '---', line)
            cleaned_lines.append(line)
        md_text = '\n'.join(cleaned_lines)
        md_text = re.sub(r'\n{3,}', '\n\n', md_text)
        return md_text.strip()

    def enhance_table_detection(self, page: pymupdf.Page) -> str:
        """Enhanced table detection and formatting."""
        text_dict = page.get_text("dict")
        blocks = text_dict["blocks"]
        table_lines = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                text = spans[0]["text"].strip()
                if not text:
                    continue
                # Detect table-like structures
                vertical_positions = [span["bbox"][1] for span in spans]
                horizontal_positions = [span["bbox"][0] for span in spans]
                # If multiple spans are vertically aligned, likely a table
                if len(spans) > 1 and abs(max(vertical_positions) - min(vertical_positions)) < 20:
                    cells = [span["text"].strip() for span in spans]
                    table_line = "| " + " | ".join(cells) + " |"
                    table_lines.append(table_line)
                else:
                    if table_lines:  # End of table
                        table_lines.append("")  # Add spacing
                    table_lines.append(text)
        return "\n".join(table_lines)

    def process_single_page(self, page_data: Tuple[int, str, str, bool, bool]) -> Tuple[int, str, Optional[str]]:
        """Process a single page with enhanced table formatting."""
        page_num, page_pdf_path, ocr_page_pdf_path, force_ocr, has_text = page_data
        try:
            # OCR processing with ocrmypdf
            ocr_args = [
                'ocrmypdf', '-l', 'eng', '--tesseract-timeout', '100',
                '--jobs', '1', '--optimize', '0', '--output-type', 'pdf'
            ]
            if not has_text and force_ocr:
                ocr_args.append('--force-ocr')
            else:
                ocr_args.append('--skip-text')
            if not has_text:
                ocr_args.extend(['--deskew', '--clean'])
            ocr_args.extend([page_pdf_path, ocr_page_pdf_path])
            result = subprocess.run(ocr_args, check=True, capture_output=True, text=True, timeout=300)
            logger.debug(f"ocrmypdf completed for page {page_num + 1}")

            # Enhanced markdown extraction
            try:
                page_markdown = pymupdf4llm.to_markdown(ocr_page_pdf_path, write_images=False, dpi=300)
                # Fallback to enhanced table detection
                fallback_doc = pymupdf.open(ocr_page_pdf_path)
                enhanced_text = self.enhance_table_detection(fallback_doc[0])
                fallback_doc.close()
                # Use enhanced if better for tables
                if '|' in enhanced_text and enhanced_text.count('|') > 2:
                    page_markdown = enhanced_text
                if page_markdown and page_markdown.strip():
                    return page_num, f"# Page {page_num + 1}\n\n{page_markdown}\n\n---\n\n", None
                else:
                    return page_num, "", None
            except Exception as e:
                # Fallback extraction
                fallback_doc = pymupdf.open(ocr_page_pdf_path)
                enhanced_text = self.enhance_table_detection(fallback_doc[0])
                fallback_doc.close()
                if enhanced_text.strip():
                    return page_num, f"# Page {page_num + 1}\n\n{enhanced_text}\n\n---\n\n", None
                else:
                    return page_num, "", None
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"ocrmypdf failed for page {page_num + 1}: {error_msg}")
            return page_num, None, f"Page {page_num + 1}: {error_msg}"
        except subprocess.TimeoutExpired:
            logger.error(f"ocrmypdf timed out for page {page_num + 1}")
            return page_num, None, f"Page {page_num + 1}: Process timed out"
        except Exception as e:
            logger.error(f"Unexpected error processing page {page_num + 1}: {str(e)}")
            return page_num, None, f"Page {page_num + 1}: {str(e)}"

    def process_pdf(self, pdf_path: str, force_ocr: bool = True) -> str:
        """Process PDF with parallel OCR and enhanced table formatting (adapted from exaOCR)."""
        start_time = time.time()
        md_text = None
        error = None
        page_count = 0
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Assume input is already PDF
                has_text = self.has_embedded_text(pdf_path)
                logger.info(f"PDF {pdf_path} has embedded text: {has_text}")

                # Split into pages
                doc = pymupdf.open(pdf_path)
                page_count = doc.page_count

                # Prepare page data
                page_data_list = []
                for page_num in range(page_count):
                    page_pdf = os.path.join(tmpdir, f"page_{page_num + 1}.pdf")
                    ocr_page_pdf = os.path.join(tmpdir, f"ocr_page_{page_num + 1}.pdf")
                    page_doc = pymupdf.open()
                    page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    page_doc.save(page_pdf)
                    page_doc.close()
                    page_data_list.append((page_num, page_pdf, ocr_page_pdf, force_ocr, has_text))
                doc.close()

                # Process pages in parallel
                page_errors = []
                all_markdown = [""] * page_count
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(page_count, 8)) as page_executor:
                    future_to_page = {
                        page_executor.submit(self.process_single_page, page_data): page_data[0]
                        for page_data in page_data_list
                    }
                    for future in concurrent.futures.as_completed(future_to_page):
                        page_num, page_markdown, page_error = future.result()
                        if page_error:
                            page_errors.append(page_error)
                            logger.warning(page_error)
                        elif page_markdown:
                            all_markdown[page_num] = page_markdown

                # Combine markdown
                if all_markdown:
                    combined_markdown = "".join(all_markdown)
                    md_text = self.clean_markdown(combined_markdown)
                if page_errors:
                    error = "; ".join(page_errors[:3])
                    if len(page_errors) > 3:
                        error += f"... and {len(page_errors) - 3} more errors"
        except Exception as e:
            logger.error(f"Unexpected error processing PDF {pdf_path}: {str(e)}")
            raise
        processing_time = time.time() - start_time
        logger.info(f"Processed PDF {pdf_path} in {processing_time:.2f} seconds with {page_count} pages")
        if error:
            raise ValueError(f"OCR errors: {error}")
        return md_text