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

def create_new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    st.session_state.chat_sessions[chat_id] = {
        "chat_id": chat_id,
        "user_id": st.session_state.get("user_id", "default_user"),
        "messages": [],
        "created_at": datetime.datetime.now().isoformat(),
        "updated_at": datetime.datetime.now().isoformat(),
        "document_ids": []
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

def get_file_preview_url(file_id: str) -> Optional[str]:
    """Generate a URL for file preview using the /preview endpoint"""
    file_meta = next(
        (f for f in st.session_state.file_metadata if f['file_id'] == file_id),
        None
    )
    if not file_meta:
        return None
    return f"http://rag-service:8000/preview/{file_id}?user_id={file_meta['user_id']}&X-API-Key={settings.openai_api_key}"

def update_selected_docs(file_id: str):
    """Callback for document selection checkboxes"""
    if file_id in st.session_state.selected_docs:
        st.session_state.selected_docs.remove(file_id)
    else:
        st.session_state.selected_docs.append(file_id)
    session_state.save()

def render_document_management(user_id: str):
    """Render the document management sidebar"""
    st.sidebar.title("Document Management")

    # File upload section
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
                        error_messages.append(f"File {uploaded_file.name} exceeds {settings.max_document_size//(1024*1024)}MB limit")
                        continue

                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"user_id": user_id}
                    try:
                        response = requests.post(
                            "http://rag-service:8000/process_file",
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
                            logger.info(f"Uploaded file: {uploaded_file.name}")
                        else:
                            error_messages.append(f"Upload failed for {uploaded_file.name}: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        error_messages.append(f"Upload error for {uploaded_file.name}: {str(e)}")

            if success_count > 0:
                st.success(f"Successfully uploaded {success_count} file(s)!")
                st.session_state.upload_key += 1
                session_state.save()
                st.rerun()
            if error_messages:
                for error in error_messages:
                    st.error(error)
            st.session_state.upload_trigger = False

    # Document list
    st.sidebar.markdown("### Uploaded Documents")
    try:
        with st.spinner("Loading documents..."):
            response = requests.get(
                f"http://rag-service:8000/documents?user_id={user_id}",
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
                                    with st.spinner("Deleting..."):
                                        response = requests.delete(
                                            f"http://rag-service:8000/documents/{file['file_id']}?user_id={user_id}",
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
                                        logger.info(f"Deleted document: {file['filename']}")
                                        st.rerun()
                                    else:
                                        st.error(f"Delete failed: {response.json().get('detail', 'Unknown error')}")
                                except Exception as e:
                                    st.error(f"Delete error: {str(e)}")
            else:
                st.sidebar.info("No documents uploaded yet.")
        else:
            st.sidebar.error(f"Failed to load documents: {response.json().get('detail', 'Unknown error')}")
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
                st.rerun()
        with cols[1]:
            if st.button(
                "❌",
                key=f"delete_chat_{chat_id}",
                on_click=delete_chat,
                args=(chat_id,)
            ):
                st.rerun()

def render_chat_interface(user_id: str):
    """Render the main chat interface"""
    st.title("RAG Chat Interface")

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

    if st.button("Send", key="send_button"):
        if not query or not user_id or not st.session_state.selected_docs:
            if not query and not st.session_state.selected_docs:
                st.error("Please enter a query and select at least one document.")
            elif not query:
                st.error("Please enter a query.")
            elif not st.session_state.selected_docs:
                st.error("Please select at least one document.")
        else:
            with st.spinner("Processing query..."):
                logger.debug(f"Sending query with file_ids: {st.session_state.selected_docs}")
                try:
                    payload = {
                        "query": query,
                        "user_id": user_id,
                        "file_ids": st.session_state.selected_docs,
                        "chat_id": st.session_state.current_chat_id
                    }
                    response = requests.post(
                        "http://rag-service:8000/chat",
                        data=payload,
                        headers={"X-API-Key": settings.openai_api_key}
                    )

                    if response.status_code == 200:
                        result = response.json()
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
                            current_chat['document_ids'] = st.session_state.selected_docs
                            session_state.save()

                        # Display response
                        st.write(f"**Query**: {query}")
                        st.write(f"**Response**: {answer}")

                        if result.get("sources"):
                            st.subheader("Sources")
                            for source in result["sources"]:
                                st.write(f"- **{source['filename']}** (Section {source['chunk_index']})")
                    else:
                        st.error(f"Chat failed: {response.json().get('detail', 'Unknown error')}")
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
    """Render the knowledge graph visualization using pyvis"""
    logger.info(f"Rendering knowledge graph for user_id={user_id}, file_id={file_id}")
    st.markdown("### Knowledge Graph")
    try:
        with st.spinner("Loading knowledge graph..."):
            url = f"http://rag-service:8000/knowledge_graph?user_id={user_id}"
            if file_id:
                url += f"&file_id={file_id}"
            logger.debug(f"Requesting knowledge graph from: {url}")
            response = requests.get(
                url,
                headers={"X-API-Key": settings.openai_api_key}
            )

        if response.status_code == 200:
            graph_data = response.json()
            logger.debug(f"Graph data received: {json.dumps(graph_data, indent=2)}")
            if not graph_data["nodes"]:
                st.warning("No knowledge graph data available for this user or document.")
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

            # Save to temporary HTML file
            temp_file = f"/tmp/knowledge_graph_{user_id}_{file_id if file_id else 'all'}.html"
            logger.debug(f"Saving graph to: {temp_file}")
            net.write_html(temp_file)
            with open(temp_file, "r") as f:
                html_content = f.read()
            components.html(html_content, height=600)
        else:
            st.error(f"Failed to load knowledge graph: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error loading knowledge graph: {str(e)}")
        st.error(f"Error loading knowledge graph: {str(e)}")

def render_debug_page():
    """Render the debug page for a specific file"""
    file_id = st.query_params.get("file_id", [None])[0]
    logger.info(f"Rendering debug page for file_id={file_id}, query_params={st.query_params}")
    if not file_id:
        logger.error("No file_id provided in query parameters")
        st.error("No File ID provided")
        st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))
        return

    file_meta = next(
        (f for f in st.session_state.file_metadata if f['file_id'] == file_id),
        None
    )

    if not file_meta:
        logger.error(f"File ID {file_id} not found in session state")
        st.error("File not found in session state")
        st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))
        return

    st.title(f"Debug: File ID {file_id[:8]}...")
    # File metadata section
    st.markdown("### File Metadata")
    cols = st.columns(2)
    with cols[0]:
        st.metric("Filename", file_meta['filename'])
        st.metric("File Type", file_meta['file_type'])
    with cols[1]:
        st.metric("Upload Date", file_meta['upload_date'])
        st.metric("Size", f"{file_meta['size'] / 1024:.1f} KB")

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
        with st.spinner("Querying vector database..."):
            response = requests.post(
                "http://rag-service:8000/search",
                data={
                    "query": "",
                    "user_id": file_meta['user_id'],
                    "file_id": file_id,
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
                            "user_id": file_meta['user_id'],
                            "document_id": point['document_id'],
                            "parent_section": point['parent_section'],
                            "content_preview": f"{point['content'][:200]}..." if point['content'] else "Empty",
                            "entities": point['entities'],
                            "relationships": point['relationships'],
                            "score": point['score']
                        })
            else:
                st.warning("No chunks found in vector database for this file")
        else:
            st.error(f"Failed to query chunks: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Search query failed for file_id {file_id}: {str(e)}")
        st.error(f"Failed to query chunks: {str(e)}")

    # Knowledge graph visualization section
    render_knowledge_graph(file_meta['user_id'], file_id)

    st.button("Back to Main", on_click=lambda: st.query_params.update(page="main"))

def render_main_page():
    """Render the main application page"""
    st.title("RAG Microservice Interface")
    st.warning("Note: Hard reloading (Ctrl+F5) will clear in-memory state, but documents and chats will reload from disk.")

    # Get user ID
    user_id = st.text_input(
        "Enter User ID",
        value="default_user",
        key="user_id_input"
    )
    if user_id:
        st.session_state.user_id = user_id

    # Render sidebars
    render_document_management(user_id)
    render_chat_sessions(user_id)

    # Render appropriate main content
    if st.session_state.current_chat_id:
        render_chat_interface(user_id)
    else:
        st.write("Create or select a chat session to begin.")

# Router
page = st.query_params.get("page", ["main"])[0]
logger.info(f"Query parameter 'page': {page}, full query params: {st.query_params}")
if page == "main":
    render_main_page()
elif page == "debug":
    render_debug_page()
else:
    st.error(f"Invalid page: {page}")
    st.button("Go to Main Page", on_click=lambda: st.query_params.update(page="main"))