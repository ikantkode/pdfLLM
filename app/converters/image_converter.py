import logging
import os
import tempfile
import img2pdf
from app.utils.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

def convert_to_markdown(file_path: str) -> str:
    """
    Convert an image file to markdown by converting to PDF and using OCRProcessor.
    
    Args:
        file_path (str): Path to the image file (.jpg, .jpeg, .png, .heic, .webp).
        
    Returns:
        str: Extracted markdown content.
    """
    logger.info(f"Converting image {file_path} to markdown")
    ocr_processor = OCRProcessor()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert image to PDF
            pdf_path = os.path.join(tmpdir, "temp.pdf")
            with open(file_path, "rb") as image_file:
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(img2pdf.convert(image_file))
            
            # Process PDF with OCRProcessor
            markdown_content = ocr_processor.process_pdf(pdf_path)
            logger.info(f"Successfully converted image {file_path} to markdown")
            return markdown_content
    except Exception as e:
        logger.error(f"Failed to convert image {file_path} to markdown: {str(e)}")
        raise ValueError(f"Image conversion failed: {str(e)}")