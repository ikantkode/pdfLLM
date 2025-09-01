import logging
import pandas as pd
import os
import magic

logger = logging.getLogger(__name__)

def convert_to_markdown(file_path: str) -> str:
    """
    Convert an Excel or CSV file to markdown table format.
    
    Args:
        file_path (str): Path to the Excel/CSV file (.csv, .xls, .xlsx, .ods).
        
    Returns:
        str: Markdown table content.
    """
    logger.info(f"Converting Excel/CSV {file_path} to markdown")
    
    try:
        # Detect actual file type
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path, engine='openpyxl')
        elif file_path.lower().endswith('.xls'):
            if 'html' in file_type.lower() or 'xml' in file_type.lower():
                # Handle HTML/XML disguised as .xls
                try:
                    dfs = pd.read_html(file_path)
                    if dfs:
                        df = dfs[0]  # Use the first table found
                    else:
                        raise ValueError("No tables found in HTML/XML file")
                except Exception as e:
                    logger.error(f"Failed to process .xls file (likely HTML/XML): {str(e)}")
                    raise ValueError(f"Failed to process .xls file: {str(e)}")
            else:
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except Exception as e:
                    logger.error(f"Unsupported or corrupt .xls file: {str(e)}")
                    raise ValueError(f"Unsupported or corrupt .xls file: {str(e)}")
        else:
            logger.error(f"Unsupported file format: {file_path}")
            raise ValueError("Unsupported file format")
        
        # Convert to markdown table
        markdown = "# Excel/CSV Content\n\n"
        markdown += df.to_markdown(index=False)
        logger.info(f"Successfully converted Excel/CSV {file_path} to markdown")
        return markdown
    except Exception as e:
        logger.error(f"Failed to convert Excel/CSV {file_path} to markdown: {str(e)}")
        raise ValueError(f"Excel/CSV conversion failed: {str(e)}")