from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
from PIL import Image, ImageEnhance
from io import StringIO
import pandas as pd
from tabulate import tabulate
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def correct_image_orientation(image):
    try:
        # Detect orientation using Tesseract
        osd = pytesseract.image_to_osd(image)
        rotation_angle = 0
        for line in osd.split('\n'):
            if 'Rotate' in line:
                rotation_angle = int(line.split(':')[1].strip())
        if rotation_angle != 0:
            image = image.rotate(-rotation_angle, expand=True)
        return image
    except Exception as e:
        logger.error(f"Orientation detection failed: {e}")
        return image

def normalize_text(text):
    """Normalize text to fix common OCR formatting issues"""
    # Fix spaced numbers (e.g., "40 , 366 . 61" → "40366.61")
    text = re.sub(r'(\d)\s*,\s*(\d{3})\s*\.\s*(\d{2})', r'\1\2.\3', text)
    # Fix numbers with spaces (e.g., "14 , 200" → "14200")
    text = re.sub(r'(\d)\s*,\s*(\d{3})', r'\1\2', text)
    # Fix spaces between characters in words (e.g., "D z i u b a" → "Dziuba")
    text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
    # Fix payroll-specific formats (e.g., "1,510.88Net1,143.20" → "Gross $1,510.88, Net $1,143.20")
    text = re.sub(r'(\d+\.\d{2})Net(\d+\.\d{2})', r'Gross $\1, Net $\2', text)
    text = re.sub(r'(\d+\.\d{2})\s*perhour', r'$\1 per hour', text, flags=re.IGNORECASE)
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    # Remove extra spaces around punctuation
    text = re.sub(r'\s*([,.])\s*', r'\1', text)
    return text.strip()

def detect_table(text):
    """Detect and structure table-like data from text"""
    lines = text.split('\n')
    table_data = []
    headers = None
    for line in lines:
        # Split on spaces, but preserve multi-word names
        cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
        if not cells:
            continue
        if not headers and len(cells) >= 4:  # Assume headers if row has multiple columns
            headers = cells
            table_data.append(headers)
        elif headers and len(cells) >= len(headers) - 1:  # Allow for slightly irregular rows
            table_data.append(cells[:len(headers)])
    return table_data, headers

def convert_to_markdown(file_path):
    try:
        # Try to extract text directly
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        if text.strip():
            # Normalize extracted text
            text = normalize_text(text)
            # Attempt to parse table-like data
            try:
                table_data, headers = detect_table(text)
                if table_data and headers:
                    # Ensure consistent column count
                    max_cols = len(headers)
                    for row in table_data:
                        while len(row) < max_cols:
                            row.append("")
                    df = pd.DataFrame(table_data[1:], columns=headers)
                    markdown = f"# PDF Content\n\n{tabulate(df, headers='keys', tablefmt='pipe')}\n\n## Raw Text\n\n{text}"
                else:
                    markdown = f"# PDF Content\n\n{text}"
            except Exception as e:
                logger.warning(f"Table parsing failed: {e}")
                markdown = f"# PDF Content\n\n{text}"
            return markdown
        
        # If no text extracted, treat as scanned PDF
        images = convert_from_path(file_path, dpi=300)  # Increased DPI for better OCR
        markdown = "# Scanned PDF Content\n\n"
        
        for i, image in enumerate(images):
            # Correct orientation
            image = correct_image_orientation(image)
            # Enhance contrast for better OCR
            image = ImageEnhance.Contrast(image).enhance(2.0)
            # Perform OCR
            text = pytesseract.image_to_string(image)
            # Normalize OCR output
            text = normalize_text(text)
            # Attempt table detection
            table_data, headers = detect_table(text)
            if table_data and headers:
                max_cols = len(headers)
                for row in table_data:
                    while len(row) < max_cols:
                        row.append("")
                df = pd.DataFrame(table_data[1:], columns=headers)
                markdown += f"## Page {i+1}\n\n{tabulate(df, headers='keys', tablefmt='pipe')}\n\n### Raw Text\n\n{text}\n\n"
            else:
                markdown += f"## Page {i+1}\n\n{text}\n\n"
        
        return markdown
    
    except Exception as e:
        logger.error(f"Could not process PDF: {str(e)}")
        return f"# Error\n\nCould not process PDF: {str(e)}"