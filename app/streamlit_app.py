import streamlit as st
from converters import image_converter, doc_converter, excel_converter, pdf_converter, txt_converter
from utils.qdrant_handler import QdrantHandler
from utils.text_processor import TextProcessor
import os
import uuid
import datetime
import logging
import base64
import openai
from io import BytesIO
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    'images': ['.heic', '.jpg', '.jpeg', '.png'],
    'documents': ['.doc', '.docx'],
    'spreadsheets': ['.xls', '.xlsx', '.csv'],
    'pdfs': ['.pdf'],
    'text': ['.txt']
}

# State storage
STATE_FILE = "/app/data/state.json"

def save_state():
    """Save file_metadata and chat_sessions to JSON"""
    state = {
        "file_metadata": st.session_state.file_metadata,
        "chat_sessions": st.session_state.chat_sessions,
        "selected_docs": st.session_state.selected_docs,
        "current_chat_id": st.session_state.current_chat_id
    }
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        logger.info("State saved to state.json")
    except Exception as e:
        logger.error(f"Failed to save state: {str(e)}")

def load_state():
    """Load file_metadata and chat_sessions from JSON"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
                st.session_state.file_metadata = state.get("file_metadata", [])
                st.session_state.chat_sessions = state.get("chat_sessions", {})
                st.session_state.selected_docs = state.get("selected_docs", [])
                st.session_state.current_chat_id = state.get("current_chat_id", None)
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")

# Initialize session state
def initialize_session_state():
    if "file_metadata" not in st.session_state:
        st.session_state.file_metadata = []
    if "file_content_map" not in st.session_state:
        st.session_state.file_content_map = {}
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "selected_docs" not in st.session_state:
        st.session_state.selected_docs = []
    if "upload_trigger" not in st.session_state:
        st.session_state.upload_trigger = False
    load_state()

initialize_session_state()

# Initialize Qdrant connection
@st.cache_resource
def get_qdrant_handler():
    try:
        return QdrantHandler(host="qdrant", port=6333, collection_name="rag_chunks")
    except Exception as e:
        logger.error(f"Qdrant connection failed: {str(e)}")
        st.error("Failed to connect to Qdrant. Please check the connection.")
        st.stop()

qdrant_handler = get_qdrant_handler()

# Initialize TextProcessor
@st.cache_resource
def get_text_processor():
    try:
        return TextProcessor()
    except Exception as e:
        logger.error(f"TextProcessor initialization failed: {str(e)}")
        st.warning("Text processing limited - proceeding without embeddings")
        return None

text_processor = get_text_processor()

def get_file_converter(file_ext):
    """Return the appropriate converter for the file extension"""
    if file_ext in SUPPORTED_EXTENSIONS['images']:
        return image_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['documents']:
        return doc_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['spreadsheets']:
        return excel_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['pdfs']:
        return pdf_converter.convert_to_markdown
    elif file_ext in SUPPORTED_EXTENSIONS['text']:
        return txt_converter.convert_to_markdown
    return None

def process_uploaded_file(uploaded_file, user_id):
    """Handle file upload and processing"""
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    file_id = str(uuid.uuid4())
    output_dir = "temp_uploads"
    os.makedirs(output_dir, exist_ok=True)
    temp_path = f"{output_dir}/{file_id}{file_ext}"
    
    try:
        # Save uploaded file
        file_content = uploaded_file.getbuffer()
        with open(temp_path, "wb") as f:
            f.write(file_content)
        
        # Get appropriate converter
        converter = get_file_converter(file_ext)
        if not converter:
            st.error(f"Unsupported file format: {file_ext}")
            return None

        with st.spinner(f"Processing {uploaded_file.name}..."):
            markdown_content = converter(temp_path)
            
            # Process chunks if text processor is available
            if text_processor:
                try:
                    cleaned_markdown = text_processor.clean_markdown(markdown_content)
                    chunks = text_processor.chunk_text(cleaned_markdown)
                    chunks = text_processor.generate_embeddings(chunks)
                    
                    for chunk in chunks:
                        try:
                            chunk['document_id'] = file_id
                            qdrant_handler.save_chunk(chunk, user_id)
                        except Exception as e:
                            logger.error(f"Chunk save failed for {file_id}: {str(e)}")
                            continue
                except Exception as e:
                    logger.error(f"Processing failed for {file_id}: {str(e)}")
                    st.warning(f"Content processing failed for {uploaded_file.name} - saved raw markdown only")
            
            # Store file metadata
            if not any(f['file_id'] == file_id for f in st.session_state.file_metadata):
                st.session_state.file_metadata.append({
                    "file_id": file_id,
                    "filename": uploaded_file.name,
                    "file_type": file_ext,
                    "upload_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "content": base64.b64encode(file_content).decode(),
                    "markdown_content": markdown_content,
                    "user_id": user_id
                })
                st.session_state.file_content_map[file_id] = file_content
                if file_id not in st.session_state.selected_docs:
                    st.session_state.selected_docs.append(file_id)
                save_state()
                logger.info(f"Processed file: {uploaded_file.name} (ID: {file_id})")
            
            st.success(f"Document processed: {uploaded_file.name} (ID: {file_id})")
            return file_id
            
    except Exception as e:
        logger.error(f"File processing error for {uploaded_file.name}: {str(e)}")
        st.error(f"Failed to process file: {str(e)}")
        return None
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"Temp file cleanup failed for {temp_path}: {str(e)}")

def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = []
    st.session_state.current_chat_id = chat_id
    save_state()

def delete_chat(chat_id):
    """Delete a chat session"""
    del st.session_state.chat_sessions[chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None
    save_state()

def get_file_preview_url(file_id):
    """Generate a data URL for file preview"""
    file_meta = next((f for f in st.session_state.file_metadata if f['file_id'] == file_id), None)
    if not file_meta:
        return None
        
    mime_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".heic": "image/heic",
        ".doc": "application/msword",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xls": "application/vnd.ms-excel",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".csv": "text/csv",
        ".txt": "text/plain"
    }
    
    mime_type = mime_map.get(file_meta['file_type'], "application/octet-stream")
    file_content = base64.b64decode(file_meta['content'])
    b64_content = base64.b64encode(file_content).decode()
    return f"data:{mime_type};base64,{b64_content}"

def update_selected_docs(file_id):
    """Callback for document selection checkboxes"""
    if file_id in st.session_state.selected_docs:
        st.session_state.selected_docs.remove(file_id)
    else:
        st.session_state.selected_docs.append(file_id)
    save_state()

def render_main_page():
    """Render the main chat interface"""
    st.title("RAG Microservice Debugger")
    st.warning("Note: Hard reloading (Ctrl+F5) will clear in-memory state, but documents and chats will reload from disk.")
    
    # Sidebar for file upload and document list
    with st.sidebar:
        st.title("Document Management")
        user_id = st.text_input("Enter User ID", value="test_user", key="user_id_input")
        
        # File upload section
        uploaded_file = st.file_uploader(
            "Upload a file", 
            type=(
                SUPPORTED_EXTENSIONS['images'] + 
                SUPPORTED_EXTENSIONS['documents'] + 
                SUPPORTED_EXTENSIONS['spreadsheets'] + 
                SUPPORTED_EXTENSIONS['pdfs'] + 
                SUPPORTED_EXTENSIONS['text']
            ),
            accept_multiple_files=False,
            key=f"file_uploader_{st.session_state.get('upload_key', 0)}"
        )
        
        if uploaded_file and not st.session_state.upload_trigger:
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("File size exceeds 200MB limit")
            else:
                st.session_state.upload_trigger = True
                file_id = process_uploaded_file(uploaded_file, user_id)
                if file_id:
                    st.session_state.upload_key = st.session_state.get('upload_key', 0) + 1
                    st.session_state.upload_trigger = False
                    st.rerun()
        
        # Document list
        st.markdown("### Uploaded Documents")
        if st.session_state.file_metadata:
            for file in st.session_state.file_metadata:
                if file.get("user_id") == user_id:
                    with st.container():
                        cols = st.columns([3, 1, 1, 1])
                        with cols[0]:
                            st.write(f"**{file['filename']}**")
                            st.caption(f"{file['file_type']} - {file['upload_date']}")
                        
                        # Preview button
                        with cols[1]:
                            preview_url = get_file_preview_url(file['file_id'])
                            if preview_url:
                                st.markdown(
                                    f'<a href="{preview_url}" target="_blank" style="text-decoration: none;">🔍</a>',
                                    unsafe_allow_html=True
                                )
                        
                        # Debug button
                        with cols[2]:
                            st.markdown(
                                f'<a href="?page=debug&file_id={file["file_id"]}" target="_blank" style="text-decoration: none;">🐞</a>',
                                unsafe_allow_html=True
                            )
                        
                        # Delete button
                        with cols[3]:
                            if st.button("🗑️", key=f"delete_{file['file_id']}"):
                                try:
                                    qdrant_handler.delete_by_document_id(file['file_id'])
                                    st.session_state.file_metadata = [
                                        f for f in st.session_state.file_metadata 
                                        if f['file_id'] != file['file_id']
                                    ]
                                    if file['file_id'] in st.session_state.selected_docs:
                                        st.session_state.selected_docs.remove(file['file_id'])
                                    st.session_state.file_content_map.pop(file['file_id'], None)
                                    save_state()
                                    logger.info(f"Deleted document: {file['filename']} (ID: {file['file_id']})")
                                    st.rerun()
                                except Exception as e:
                                    logger.error(f"Failed to delete document {file['file_id']}: {str(e)}")
                                    st.error(f"Failed to delete document: {str(e)}")
        
        # Chat sessions
        st.markdown("### Chat Sessions")
        if st.button("+ New Chat", key="new_chat_button"):
            create_new_chat()
            st.rerun()
        
        for chat_id in list(st.session_state.chat_sessions.keys()):
            cols = st.columns([4, 1])
            with cols[0]:
                if st.button(
                    f"💬 {chat_id[:8]}...", 
                    key=f"chat_{chat_id}",
                    use_container_width=True
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()
            with cols[1]:
                if st.button(
                    "❌", 
                    key=f"delete_chat_{chat_id}",
                    on_click=delete_chat,
                    args=(chat_id,)
                ):
                    st.rerun()
    
    # Main chat interface
    if st.session_state.current_chat_id:
        # Document selection for context
        if st.session_state.file_metadata:
            st.markdown("### Select Documents for Context")
            cols = st.columns(3)
            for i, file in enumerate(st.session_state.file_metadata):
                if file.get("user_id") == user_id:
                    with cols[i % 3]:
                        st.checkbox(
                            file['filename'],
                            key=f"doc_select_{file['file_id']}",
                            value=file['file_id'] in st.session_state.selected_docs,
                            on_change=update_selected_docs,
                            args=(file['file_id'],)
                        )
        
        # Chat input
        st.markdown("### Chat")
        query = st.text_area("Enter your query", height=100, key="query_input")
        
        if st.button("Send", key="send_button") and query and user_id:
            with st.spinner("Searching documents..."):
                try:
                    # Generate query embedding
                    query_embedding = openai.Embedding.create(
                        model="text-embedding-3-small",
                        input=query
                    )['data'][0]['embedding']
                    
                    # Search Qdrant with selected documents filter
                    filters = {"must": [{"key": "user_id", "match": {"value": user_id}}]}
                    if st.session_state.selected_docs:
                        filters["must"].append({"key": "document_id", "match": {"any": st.session_state.selected_docs}})
                    
                    results = qdrant_handler.client.search(
                        collection_name="rag_chunks",
                        query_vector=query_embedding,
                        query_filter=filters,
                        limit=5
                    )
                    
                    if results:
                        # Prepare context
                        file_map = {f['file_id']: f['filename'] for f in st.session_state.file_metadata}
                        context = "\n\n".join([
                            f"📄 **{file_map.get(r.payload.get('document_id', 'Unknown'), 'Unknown')}** "
                            f"(Section {r.payload.get('chunk_index', 'N/A')}):\n"
                            f"{r.payload.get('content', '')}\n"
                            for r in results
                        ])
                        
                        # Stream response
                        prompt = (
                            f"Context:\n{context}\n\n"
                            f"Query: {query}\n\n"
                            "Answer the query based on the provided context. "
                            "Cite the document name and section where information is sourced."
                        )
                        
                        response_container = st.empty()
                        full_response = ""
                        
                        with st.spinner("Generating response..."):
                            response_stream = openai.ChatCompletion.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "system", 
                                        "content": "You are a helpful assistant that answers queries "
                                                "based on document context, citing sources clearly."
                                    },
                                    {"role": "user", "content": prompt}
                                ],
                                max_tokens=500,
                                stream=True
                            )
                            
                            for chunk in response_stream:
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {}).get('content', '')
                                    full_response += delta
                                    response_container.markdown(full_response + "▌")
                        
                        # Store in chat history
                        st.session_state.chat_sessions[st.session_state.current_chat_id].append({
                            "query": query,
                            "response": full_response,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        save_state()
                        logger.info(f"Chat response generated for query: {query}")
                        st.rerun()
                    
                    else:
                        st.warning("No relevant chunks found for the query.")
                
                except Exception as e:
                    logger.error(f"Chat processing failed: {str(e)}")
                    st.error(f"Failed to process query: {str(e)}")
        
        # Display chat history
        st.markdown("### Chat History")
        for chat in st.session_state.chat_sessions.get(st.session_state.current_chat_id, []):
            st.markdown(f"**🗣️ User** ({chat['timestamp']}):")
            st.markdown(f"> {chat['query']}")
            st.markdown(f"**🤖 AI**:")
            st.markdown(chat['response'])
            st.markdown("---")
    
    else:
        st.write("Create a new chat session to start chatting.")

def render_debug_page():
    """Render the debug page for a specific file"""
    file_id = st.query_params.get("file_id", [None])[0]
    if not file_id:
        st.error("No File ID provided")
        st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))
        return
    
    st.title(f"Debug: File ID {file_id[:8]}...")
    file_meta = next((f for f in st.session_state.file_metadata if f['file_id'] == file_id), None)
    
    if not file_meta:
        st.error("File not found in session state")
        st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))
        return
    
    # File metadata section
    st.markdown("### File Metadata")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Filename", file_meta['filename'])
        st.metric("File Type", file_meta['file_type'])
    with cols[1]:
        st.metric("Upload Date", file_meta['upload_date'])
        st.metric("Size", f"{len(base64.b64decode(file_meta['content'])) / 1024:.1f} KB")
    
    # Content preview section
    with st.expander("Content Preview"):
        if file_meta.get('markdown_content'):
            st.text_area(
                "Markdown Content",
                value=file_meta['markdown_content'],
                height=300,
                disabled=True
            )
        else:
            st.warning("No content preview available")
    
    # Vector database chunks section
    st.markdown("### Vector Database Chunks")
    try:
        points, _ = qdrant_handler.client.scroll(
            collection_name="rag_chunks",
            scroll_filter={"must": [{"key": "document_id", "match": {"value": file_id}}]},
            limit=50,
            with_payload=True,
            with_vectors=False
        )
        
        if points:
            for point in points:
                with st.expander(f"Chunk {point.payload.get('chunk_index', 'N/A')}"):
                    st.json({
                        "id": str(point.id),
                        "user_id": point.payload.get("user_id"),
                        "document_id": point.payload.get("document_id"),
                        "parent_section": point.payload.get("parent_section"),
                        "content_preview": f"{point.payload.get('content', '')[:200]}..." 
                            if point.payload.get('content') else "Empty",
                        "metadata": {k: v for k in point.payload 
                                   if k not in ["content", "user_id", "document_id"]}
                    })
        else:
            st.warning("No chunks found in vector database for this file")
            
    except Exception as e:
        logger.error(f"Qdrant query failed for file_id {file_id}: {str(e)}")
        st.error(f"Failed to query Qdrant: {str(e)}")
    
    st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))

# Router
page = st.query_params.get("page", ["main"])[0]

if page == "main":
    render_main_page()
elif page == "debug":
    render_debug_page()
else:
    st.error("Invalid page")
    st.button("Go to Main Page", on_click=lambda: st.query_params.update(page="main"))