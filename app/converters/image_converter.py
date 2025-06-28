from PIL import Image
import pytesseract
import os

def convert_to_markdown(file_path):
    # Convert to JPG if not already
    if file_path.lower().endswith('.heic'):
        image = Image.open(file_path).convert("RGB")
        jpg_path = file_path.replace('.heic', '.jpg')
        image.save(jpg_path, "JPEG")
        file_path = jpg_path
    
    # Perform OCR
    text = pytesseract.image_to_string(Image.open(file_path))
    
    # Convert to markdown (basic formatting)
    markdown = f"# Extracted Text from Image\n\n{text}"
    
    # Clean up if we converted to JPG
    if file_path.endswith('.jpg') and 'heic' in file_path.lower():
        os.remove(file_path)
    
    return markdown