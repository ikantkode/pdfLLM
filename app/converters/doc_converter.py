from docx import Document
import os

def convert_to_markdown(file_path):
    doc = Document(file_path)
    markdown = "# Document Content\n\n"
    
    for para in doc.paragraphs:
        if para.text.strip():
            markdown += f"{para.text}\n\n"
    
    return markdown