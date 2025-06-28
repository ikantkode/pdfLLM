from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
from PIL import Image

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
        print(f"Orientation detection failed: {e}")
        return image

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
            return f"# PDF Content\n\n{text}"
        
        # If no text extracted, treat as scanned PDF
        images = convert_from_path(file_path)
        markdown = "# Scanned PDF Content\n\n"
        
        for i, image in enumerate(images):
            # Correct orientation
            image = correct_image_orientation(image)
            # Perform OCR
            text = pytesseract.image_to_string(image)
            markdown += f"## Page {i+1}\n\n{text}\n\n"
        
        return markdown
    
    except Exception as e:
        return f"# Error\n\nCould not process PDF: {str(e)}"