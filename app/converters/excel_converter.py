import pandas as pd
import os
import magic  # For file type detection

def convert_to_markdown(file_path):
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
                raise ValueError(f"Failed to process .xls file (likely HTML/XML): {str(e)}")
        else:
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                raise ValueError(f"Unsupported or corrupt .xls file: {str(e)}")
    else:
        raise ValueError("Unsupported file format")
    
    # Convert to markdown table
    markdown = "# Excel/CSV Content\n\n"
    markdown += df.to_markdown(index=False)
    
    return markdown