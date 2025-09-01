import logging
import os
import subprocess
import tempfile
from app.utils.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

def convert_to_markdown(file_path: str) -> str:
    """
    Convert a document file (.doc, .docx, .odt) to markdown by converting to PDF and using OCRProcessor.
    
    Args:
        file_path (str): Path to the document file.
        
    Returns:
        str: Extracted markdown content.
    """
    logger.info(f"Converting document {file_path} to markdown")
    ocr_processor = OCRProcessor()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Convert document to PDF using LibreOffice
            pdf_path = os.path.join(tmpdir, "temp.pdf")
            result = subprocess.run([
                'libreoffice', '--headless', '--convert-to', 'pdf',
                '--outdir', tmpdir, file_path
            ], check=True, capture_output=True, text=True, timeout=300)
            
            converted_pdf_name = os.path.splitext(os.path.basename(file_path))[0] + '.pdf'
            pdf_path = os.path.join(tmpdir, converted_pdf_name)
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"Converted PDF not found: {pdf_path}")
            
            # Process PDF with OCRProcessor
            markdown_content = ocr_processor.process_pdf(pdf_path)
            logger.info(f"Successfully converted document {file_path} to markdown")
            return markdown_content
    except subprocess.CalledProcessError as e:
        logger.error(f"LibreOffice conversion failed for {file_path}: {e.stderr}")
        raise ValueError(f"Document conversion failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Failed to convert document {file_path} to markdown: {str(e)}")
        raise ValueError(f"Document conversion failed: {str(e)}")