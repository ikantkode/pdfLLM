from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
from PIL import Image
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
    """Normalize text to fix common formatting issues, especially numbers."""
    # Fix spaced numbers (e.g., "40 , 366 . 61" -> "40366.61")
    text = re.sub(r'(\d)\s*,\s*(\d{3})\s*\.\s*(\d{2})', r'\1\2.\3', text)
    # Fix numbers with spaces (e.g., "14 , 200" -> "14200")
    text = re.sub(r'(\d)\s*,\s*(\d{3})', r'\1\2', text)
    # Remove extra spaces around punctuation
    text = re.sub(r'\s*([,.])\s*', r'\1', text)
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
                # Split text into lines and detect potential table structure
                lines = text.split('\n')
                table_data = [line.split() for line in lines if line.strip()]
                if len(table_data) > 1 and len(set(len(row) for row in table_data)) == 1:
                    # Assume table structure if rows have consistent column counts
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
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
            from PIL import ImageEnhance
            image = ImageEnhance.Contrast(image).enhance(2.0)
            # Perform OCR
            text = pytesseract.image_to_string(image)
            # Normalize OCR output
            text = normalize_text(text)
            markdown += f"## Page {i+1}\n\n{text}\n\n"
        
        return markdown
    
    except Exception as e:
        logger.error(f"Could not process PDF: {str(e)}")
        return f"# Error\n\nCould not process PDF: {str(e)}"