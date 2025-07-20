import streamlit as st
import requests
import json
import base64
import re
import uuid
import os
import datetime
import logging
from io import BytesIO
from typing import Dict, List, Optional
from pathlib import Path
from config import settings
from pyvis.network import Network
import streamlit.components.v1 as components

# Set up logging
Path(settings.data_dir).joinpath("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(settings.data_dir) / "logs" / "streamlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
STATE_FILE = Path(settings.data_dir) / "streamlit_state.json"
FASTAPI_HOST = "rag-service"
FASTAPI_PORT = "8000"

class SessionState:
    """Manage Streamlit session state with persistence"""
    def __init__(self):
        self.state_file = STATE_FILE
        self._initialize_state()

    def _initialize_state(self):
        """Initialize or load session state"""
        if "file_metadata" not in st.session_state:
            st.session_state.file_metadata = []
        if "chat_sessions" not in st.session_state:
            st.session_state.chat_sessions = {}
        if "current_chat_id" not in st.session_state:
            st.session_state.current_chat_id = None
        if "selected_docs" not in st.session_state:
            st.session_state.selected_docs = []
        if "upload_key" not in st.session_state:
            st.session_state.upload_key = 0
        self.load()

    def save(self):
        """Save session state to file"""
        state = {
            "file_metadata": st.session_state.file_metadata,
            "chat_sessions": st.session_state.chat_sessions,
            "selected_docs": st.session_state.selected_docs,
            "current_chat_id": st.session_state.current_chat_id
        }
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(state, f)
            logger.info("State saved successfully")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def load(self):
        """Load session state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                    st.session_state.file_metadata = state.get("file_metadata", [])
                    st.session_state.chat_sessions = state.get("chat_sessions", {})
                    st.session_state.selected_docs = state.get("selected_docs", [])
                    st.session_state.current_chat_id = state.get("current_chat_id", None)
                logger.info("State loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load state: {str(e)}")

session_state = SessionState()

def update_selected_docs():
    """Update selected documents based on checkbox states"""
    selected = []
    for file in st.session_state.file_metadata:
        checkbox_key = f"doc_select_{file['file_id']}"
        if st.session_state.get(checkbox_key, False):
            selected.append(file['file_id'])
    st.session_state.selected_docs = selected
    session_state.save()

def handle_select_all():
    """Select all documents checkbox handler"""
    for file in st.session_state.file_metadata:
        st.session_state[f"doc_select_{file['file_id']}"] = True
    update_selected_docs()

def handle_clear_all():
    """Clear all documents checkbox handler"""
    for file in st.session_state.file_metadata:
        st.session_state[f"doc_select_{file['file_id']}"] = False
    update_selected_docs()

def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "chat_id": chat_id,
        "user_id": st.session_state.get("user_id", "default_user"),
        "messages": [],
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
        "document_ids": st.session_state.selected_docs.copy()
    }
    st.session_state.current_chat_id = chat_id
    session_state.save()
    st.rerun()

def delete_chat(chat_id: str):
    """Delete a chat session"""
    if chat_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None
    session_state.save()
    st.rerun()

def get_file_preview_url(filename: str) -> Optional[str]:
    """Generate a URL for file preview"""
    file_meta = next(
        (f for f in st.session_state.file_metadata if f['filename'] == filename),
        None
    )
    if not file_meta:
        return None
    file_id = file_meta['file_id']
    return f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/preview/{file_id}?user_id={file_meta['user_id']}&X-API-Key={settings.openai_api_key}"

def render_document_management(user_id: str):
    """Render the document management sidebar"""
    st.sidebar.title("Document Management")

    with st.sidebar.expander("Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=(
                settings.supported_extensions['images'] +
                settings.supported_extensions['documents'] +
                settings.supported_extensions['spreadsheets'] +
                settings.supported_extensions['pdfs'] +
                settings.supported_extensions['text']
            ),
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.upload_key}"
        )

        if uploaded_files and not st.session_state.get("upload_trigger", False):
            st.session_state.upload_trigger = True
            success_count = 0
            error_messages = []
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.size > settings.max_document_size:
                        error_messages.append(f"File {uploaded_file.name} exceeds size limit")
                        continue

                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"user_id": user_id}
                    try:
                        response = requests.post(
                            f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/process_file",
                            files=files,
                            data=data,
                            headers={"X-API-Key": settings.openai_api_key}
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.file_metadata.append({
                                "file_id": result["file_id"],
                                "filename": result["filename"],
                                "file_type": os.path.splitext(result["filename"])[1].lower(),
                                "upload_date": datetime.datetime.now().isoformat(),
                                "size": uploaded_file.size,
                                "content": base64.b64encode(uploaded_file.getvalue()).decode(),
                                "user_id": user_id
                            })
                            success_count += 1
                        else:
                            error_messages.append(f"Upload failed for {uploaded_file.name}")
                    except Exception as e:
                        error_messages.append(f"Upload error: {str(e)}")

            if success_count > 0:
                st.success(f"Uploaded {success_count} file(s)")
                st.session_state.upload_key += 1
                session_state.save()
                st.rerun()
            if error_messages:
                for error in error_messages:
                    st.error(error)
            st.session_state.upload_trigger = False

    st.sidebar.markdown("### Uploaded Documents")
    try:
        with st.spinner("Loading documents..."):
            response = requests.get(
                f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/documents?user_id={user_id}",
                headers={"X-API-Key": settings.openai_api_key}
            )

        if response.status_code == 200:
            documents = response.json().get("documents", [])
            if documents:
                for file in documents:
                    with st.sidebar.container():
                        cols = st.columns([3, 1, 1, 1])
                        with cols[0]:
                            st.write(f"**{file['filename']}**")
                            st.caption(f"{file['file_type']} - {file['upload_date']} - {file['size']/1024:.1f} KB")

                        with cols[1]:
                            preview_url = get_file_preview_url(file['filename'])
                            if preview_url:
                                st.markdown(
                                    f'<a href="/preview/{file["filename"]}" target="_blank" style="text-decoration: none;">🔍</a>',
                                    unsafe_allow_html=True
                                )

                        with cols[2]:
                            st.markdown(
                                f'<a href="?page=debug&file_id={file["file_id"]}" target="_self" style="text-decoration: none;">🐞</a>',
                                unsafe_allow_html=True
                            )

                        with cols[3]:
                            if st.button("🗑️", key=f"delete_{file['file_id']}"):
                                try:
                                    with st.spinner("Deleting..."):
                                        response = requests.delete(
                                            f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/documents/{file['file_id']}?user_id={user_id}",
                                            headers={"X-API-Key": settings.openai_api_key}
                                        )
                                    if response.status_code == 200:
                                        st.session_state.file_metadata = [
                                            f for f in st.session_state.file_metadata
                                            if f['file_id'] != file['file_id']
                                        ]
                                        if file['file_id'] in st.session_state.selected_docs:
                                            st.session_state.selected_docs.remove(file['file_id'])
                                        session_state.save()
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Delete error: {str(e)}")
            else:
                st.sidebar.info("No documents uploaded yet.")
    except Exception as e:
        st.sidebar.error(f"Error loading documents: {str(e)}")

def render_chat_sessions(user_id: str):
    """Render the chat sessions sidebar"""
    st.sidebar.title("Chat Sessions")

    if st.sidebar.button("+ New Chat", key="new_chat_button"):
        create_new_chat()

    for chat_id, chat_data in st.session_state.chat_sessions.items():
        if chat_data['user_id'] != user_id:
            continue

        cols = st.sidebar.columns([4, 1])
        with cols[0]:
            if st.button(
                f"💬 {chat_id[:8]}... ({len(chat_data['messages']) // 2} messages)",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                st.session_state.current_chat_id = chat_id
                st.session_state.selected_docs = chat_data['document_ids'].copy()
                st.rerun()
        with cols[1]:
            if st.button(
                "❌",
                key=f"delete_chat_{chat_id}",
                on_click=delete_chat,
                args=(chat_id,)
            ):
                pass

def render_chat_interface(user_id: str):
    """Render the main chat interface"""
    st.title("RAG Chat Interface")

    # Document selection
    if st.session_state.file_metadata:
        st.markdown("### Select Documents for Context")
        
        # Initialize selected_docs if not exists
        if 'selected_docs' not in st.session_state:
            st.session_state.selected_docs = []
        
        # Select All/Clear All buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", on_click=handle_select_all):
                pass
        with col2:
            if st.button("Clear All", on_click=handle_clear_all):
                pass
        
        st.markdown(f"**Selected Documents**: {len(st.session_state.selected_docs)}")
        
        # Display checkboxes in a scrollable container
        with st.container(height=200):
            for file in st.session_state.file_metadata:
                if file.get("user_id") == user_id:
                    checkbox_key = f"doc_select_{file['file_id']}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = file['file_id'] in st.session_state.selected_docs
                    
                    st.checkbox(
                        file['filename'],
                        value=st.session_state[checkbox_key],
                        key=checkbox_key,
                        on_change=update_selected_docs
                    )

    # Chat input
    st.markdown("### Chat")
    query = st.text_area("Enter your query", height=100, key="query_input")

    if st.button("Send", key="send_button"):
        if not query or not user_id:
            st.error("Please enter a query.")
        elif not st.session_state.selected_docs:
            st.error("Please select at least one document.")
        else:
            with st.spinner("Processing query..."):
                try:
                    # Send query to chat endpoint
                    chat_payload = {
                        "query": query,
                        "user_id": user_id,
                        "file_ids": st.session_state.selected_docs,
                        "chat_id": st.session_state.current_chat_id
                    }
                    chat_response = requests.post(
                        f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/chat",
                        data=chat_payload,
                        headers={"X-API-Key": settings.openai_api_key}
                    )
                    
                    if chat_response.status_code == 200:
                        result = chat_response.json()
                        answer = clean_response(result["response"])
                        
                        # Update chat history
                        current_chat = st.session_state.chat_sessions.get(st.session_state.current_chat_id, {})
                        if current_chat:
                            current_chat['messages'].append({
                                "role": "user",
                                "content": query,
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                            current_chat['messages'].append({
                                "role": "assistant",
                                "content": answer,
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                            current_chat['updated_at'] = datetime.datetime.now().isoformat()
                            current_chat['document_ids'] = st.session_state.selected_docs.copy()
                            session_state.save()

                        # Display response
                        st.write(f"**Query**: {query}")
                        st.write(f"**Response**: {answer}")

                        if result.get("sources"):
                            st.subheader("Sources")
                            unique_sources = {}
                            for source in result["sources"]:
                                if source['document_id'] not in unique_sources:
                                    unique_sources[source['document_id']] = source
                            for source in unique_sources.values():
                                st.write(f"- **{source['filename']}** (Section {source['chunk_index']})")
                    else:
                        st.error(f"Chat error: {chat_response.text}")
                except Exception as e:
                    st.error(f"Chat error: {str(e)}")

    # Display chat history
    st.markdown("### Chat History")
    current_chat = st.session_state.chat_sessions.get(st.session_state.current_chat_id, {})
    for msg in current_chat.get('messages', []):
        if msg['role'] == 'user':
            st.markdown(f"**🗣️ User** ({msg['timestamp']}):")
            st.markdown(f"> {msg['content']}")
        else:
            st.markdown(f"**🤖 AI**:")
            st.markdown(msg['content'])
        st.markdown("---")

def clean_response(text: str) -> str:
    """Clean response text for display"""
    text = re.sub(r'\n+', '\n', text).strip()
    text = re.sub(r'[\*\_\`\#]{2,}', '', text)
    return text

def render_knowledge_graph(user_id: str, file_id: Optional[str] = None):
    """Render the knowledge graph visualization"""
    st.markdown("### Knowledge Graph")
    try:
        with st.spinner("Loading knowledge graph..."):
            url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/knowledge_graph?user_id={user_id}"
            if file_id:
                url += f"&file_id={file_id}"
            response = requests.get(
                url,
                headers={"X-API-Key": settings.openai_api_key}
            )

        if response.status_code == 200:
            graph_data = response.json()
            if not graph_data["nodes"]:
                st.warning("No knowledge graph data available")
                return
            
            net = Network(height="600px", width="100%", directed=True, notebook=True)
            for node in graph_data["nodes"]:
                net.add_node(
                    node["id"],
                    label=node["label"],
                    color="#4CAF50" if node["type"] == "entity" else "#2196F3"
                )
            for edge in graph_data["edges"]:
                net.add_edge(edge["from"], edge["to"], title=edge["label"])

            temp_file = f"/tmp/knowledge_graph_{user_id}_{file_id if file_id else 'all'}.html"
            net.write_html(temp_file)
            with open(temp_file, "r") as f:
                html_content = f.read()
            components.html(html_content, height=600)
    except Exception as e:
        st.error(f"Error loading knowledge graph: {str(e)}")

def render_debug_page():
    """Render the debug page for a specific file"""
    file_id = st.query_params.get("file_id", [None])[0]
    if not file_id:
        st.error("No File ID provided")
        st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))
        return

    file_meta = next(
        (f for f in st.session_state.file_metadata if f['file_id'] == file_id),
        None
    )

    if not file_meta:
        st.error("File not found")
        st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))
        return

    st.title(f"Debug: {file_meta['filename']}")
    
    st.markdown("### File Metadata")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Filename", file_meta['filename'])
        st.metric("File Type", file_meta['file_type'])
    with cols[1]:
        st.metric("Upload Date", file_meta['upload_date'])
        st.metric("Size", f"{file_meta['size'] / 1024:.1f} KB")

    with st.expander("Content Preview"):
        if file_meta.get('markdown_content'):
            st.text_area(
                "Markdown Content",
                value=file_meta['markdown_content'],
                height=300,
                disabled=True
            )

    st.markdown("### Vector Database Chunks")
    try:
        with st.spinner("Querying vector database..."):
            response = requests.post(
                f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/search",
                data={
                    "query": "",
                    "user_id": file_meta['user_id'],
                    "file_ids": [file_id],
                    "limit": 50,
                    "use_graph": False
                },
                headers={"X-API-Key": settings.openai_api_key}
            )

        if response.status_code == 200:
            points = response.json().get("results", [])
            if points:
                for point in points:
                    with st.expander(f"Chunk {point['chunk_index']}"):
                        st.json({
                            "id": point['chunk_id'],
                            "content_preview": f"{point['content'][:200]}...",
                            "entities": point['entities'],
                            "relationships": point['relationships'],
                            "score": point['score']
                        })
    except Exception as e:
        st.error(f"Failed to query chunks: {str(e)}")

    render_knowledge_graph(file_meta['user_id'], file_id)
    st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))

def render_main_page():
    """Render the main application page"""
    st.title("RAG Microservice Interface")
    st.warning("Note: Hard reloading (Ctrl+F5) will clear in-memory state")

    user_id = st.text_input(
        "Enter User ID",
        value="default_user",
        key="user_id_input"
    )
    if user_id:
        st.session_state.user_id = user_id

    render_document_management(user_id)
    render_chat_sessions(user_id)

    if st.session_state.current_chat_id:
        render_chat_interface(user_id)
    else:
        st.write("Create or select a chat session to begin.")

# Router
page = st.query_params.get("page", ["main"])[0]
if page == "main":
    render_main_page()
elif page == "debug":
    render_debug_page()
else:
    st.error(f"Invalid page: {page}")
    st.button("Go to Main Page", on_click=lambda: st.query_params.update(page="main"))