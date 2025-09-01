import logging
from app.utils.ocr_processor import OCRProcessor

logger = logging.getLogger(__name__)

def convert_to_markdown(file_path: str) -> str:
    """
    Convert a PDF file to markdown using OCRProcessor.
    
    Args:
        file_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted markdown content.
    """
    logger.info(f"Converting PDF {file_path} to markdown")
    ocr_processor = OCRProcessor()
    
    try:
        markdown_content = ocr_processor.process_pdf(file_path)
        logger.info(f"Successfully converted PDF {file_path} to markdown")
        return markdown_content
    except Exception as e:
        logger.error(f"Failed to convert PDF {file_path} to markdown: {str(e)}")
        raise ValueError(f"PDF conversion failed: {str(e)}")