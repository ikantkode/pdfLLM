def convert_to_markdown(file_path):
    """Convert a .txt file to markdown format."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Wrap content in markdown for consistency
        markdown_content = f"```text\n{content}\n```"
        return markdown_content
    except Exception as e:
        raise Exception(f"Failed to convert .txt to markdown: {str(e)}")