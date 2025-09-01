import logging

logger = logging.getLogger(__name__)

def convert_to_markdown(file_path: str) -> str:
    """
    Convert a text file to markdown format.
    
    Args:
        file_path (str): Path to the text file (.txt, .md, .rtf).
        
    Returns:
        str: Markdown content.
    """
    logger.info(f"Converting text file {file_path} to markdown")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        markdown_content = f"```text\n{content}\n```"
        logger.info(f"Successfully converted text file {file_path} to markdown")
        return markdown_content
    except Exception as e:
        logger.error(f"Failed to convert text file {file_path} to markdown: {str(e)}")
        raise ValueError(f"Text file conversion failed: {str(e)}")