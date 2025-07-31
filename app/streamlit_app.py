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
from app.config import settings
from pyvis.network import Network
import streamlit.components.v1 as components
from celery import Celery
import time

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

# Validate OpenAI configuration
if not settings.openai_enabled:
    logger.error("OpenAI is not enabled. Please set OPENAI_ENABLED to true.")
    st.error("Application configuration error: OpenAI is not enabled.")
    st.stop()

# Constants
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "rag-service")
FASTAPI_PORT = os.getenv("FASTAPI_PORT", "8000")
BASE_URL = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}"
CELERY_BROKER_URL = settings.celery_broker_url

# Initialize Celery for inspection
celery_app = Celery('rag_app', broker=CELERY_BROKER_URL)

class SessionState:
    def __init__(self):
        self._initialize_state()

    def _initialize_state(self):
        logger.debug("Initializing session state")
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
            logger.debug("Initialized upload_key to 0")
        if "pending_files" not in st.session_state:
            st.session_state.pending_files = {}
        if "categories" not in st.session_state:
            st.session_state.categories = ["submittals", "payrolls", "bank_statements"]
        logger.debug(f"Session state initialized: {st.session_state}")

# Initialize session state
session_state = SessionState()

def fetch_categories(user_id: str) -> List[str]:
    """Fetch categories from the /prompts endpoint."""
    try:
        response = requests.get(
            f"{BASE_URL}/prompts?user_id={user_id}",
            headers={"X-API-Key": settings.openai_api_key},
            timeout=10
        )
        if response.status_code == 200:
            prompts = response.json().get("prompts", [])
            categories = sorted([prompt["category"] for prompt in prompts])
            logger.debug(f"Fetched {len(categories)} categories: {categories}")
            return categories
        else:
            logger.error(f"Failed to fetch categories: {response.status_code} - {response.text}")
            st.error(f"Failed to fetch categories: {response.status_code} - {response.text}")
            return st.session_state.categories
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        st.error(f"Error fetching categories: {str(e)}")
        return st.session_state.categories

def get_celery_status() -> Dict:
    """Get Celery worker and task status."""
    try:
        insp = celery_app.control.inspect()
        active_tasks = insp.active() or {}
        worker_count = len(active_tasks)
        task_count = sum(len(tasks) for tasks in active_tasks.values())
        logger.debug(f"Celery status: {worker_count} workers, {task_count} active tasks")
        return {"workers": worker_count, "tasks": task_count}
    except Exception as e:
        logger.error(f"Failed to get Celery status: {str(e)}")
        return {"workers": 0, "tasks": 0}

def update_selected_docs():
    selected = []
    for file in st.session_state.file_metadata:
        checkbox_key = f"doc_select_{file['file_id']}"
        if st.session_state.get(checkbox_key, False):
            selected.append(file['file_id'])
    st.session_state.selected_docs = selected

def handle_select_all():
    for file in st.session_state.file_metadata:
        st.session_state[f"doc_select_{file['file_id']}"] = True
    update_selected_docs()

def handle_clear_all():
    for file in st.session_state.file_metadata:
        st.session_state[f"doc_select_{file['file_id']}"] = False
    update_selected_docs()

def create_new_chat():
    st.session_state.current_chat_id = None
    st.session_state.selected_docs = []
    st.rerun()

def get_file_preview_url(file_id: str, user_id: str) -> Optional[str]:
    return f"{BASE_URL}/preview/{file_id}?user_id={user_id}&X-API-Key={settings.openai_api_key}"

def check_file_status(user_id: str, file_id: str) -> Dict:
    try:
        response = requests.get(
            f"{BASE_URL}/documents?user_id={user_id}",
            headers={"X-API-Key": settings.openai_api_key},
            timeout=10
        )
        if response.status_code == 200:
            documents = response.json().get("documents", [])
            for doc in documents:
                if doc["file_id"] == file_id:
                    return {"file_id": file_id, "status": doc["status"], "filename": doc["filename"]}
        logger.error(f"Failed to check status for file_id {file_id}: {response.status_code} - {response.text}")
        return {"file_id": file_id, "status": "unknown", "filename": "Unknown"}
    except Exception as e:
        logger.error(f"Error checking file status for {file_id}: {str(e)}")
        return {"file_id": file_id, "status": "error", "filename": "Unknown", "error": str(e)}

def render_document_management(user_id: str):
    st.sidebar.title("Document Management")

    # Fetch categories dynamically
    categories = fetch_categories(user_id)
    upload_categories = categories + ["other"]

    with st.sidebar.expander("Upload Documents", expanded=True):
        if "upload_key" not in st.session_state:
            st.session_state.upload_key = 0
            logger.warning("upload_key was not initialized, set to 0")

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
        category = st.selectbox("Category", upload_categories, index=0)

        if uploaded_files and not st.session_state.get("upload_trigger", False):
            st.session_state.upload_trigger = True
            success_count = 0
            error_messages = []
            pending_files = []
            with st.spinner("Uploading files..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.size > settings.max_document_size:
                        error_msg = f"File {uploaded_file.name} size {uploaded_file.size/(1024*1024):.2f}MB exceeds limit {settings.max_document_size/(1024*1024):.2f}MB"
                        logger.error(error_msg)
                        error_messages.append(error_msg)
                        continue

                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    data = {"user_id": user_id, "category": category if category != "other" else None}
                    try:
                        response = requests.post(
                            f"{BASE_URL}/process_file",
                            files=files,
                            data=data,
                            headers={"X-API-Key": settings.openai_api_key},
                            timeout=30
                        )
                        if response.status_code == 200:
                            result = response.json()
                            pending_files.append({
                                "file_id": result["file_id"],
                                "filename": result["filename"],
                                "status": "pending"
                            })
                            success_count += 1
                        else:
                            error_msg = f"Upload failed for {uploaded_file.name}: {response.status_code} - {response.text}"
                            logger.error(error_msg)
                            error_messages.append(error_msg)
                    except requests.exceptions.RequestException as e:
                        error_msg = f"Network error uploading {uploaded_file.name}: {str(e)}"
                        logger.error(error_msg)
                        error_messages.append(error_msg)
                    except Exception as e:
                        error_msg = f"Unexpected error uploading {uploaded_file.name}: {str(e)}"
                        logger.error(error_msg)
                        error_messages.append(error_msg)

            if success_count > 0:
                st.success(f"Uploaded {success_count} file(s). Processing in background...")
                st.session_state.pending_files.update({f["file_id"]: f for f in pending_files})
                st.session_state.upload_key += 1
                st.session_state.categories = fetch_categories(user_id)
            if error_messages:
                for error in error_messages:
                    st.error(error)
            st.session_state.upload_trigger = False

    # Display Celery worker and task status
    st.sidebar.markdown("### Processing Status")
    celery_status = get_celery_status()
    st.sidebar.write(f"Active Workers: {celery_status['workers']}")
    st.sidebar.write(f"Files Processing: {celery_status['tasks']}")

    # Poll status for pending files with auto-refresh
    if st.session_state.pending_files:
        with st.sidebar.expander("File Processing Status", expanded=True):
            status_container = st.empty()
            refresh_interval = 5  # Seconds between auto-refreshes
            for _ in range(60):  # Limit to 5 minutes to prevent infinite loop
                if not st.session_state.pending_files:
                    break
                with status_container.container():
                    for file_id, file_info in list(st.session_state.pending_files.items()):
                        status_info = check_file_status(user_id, file_id)
                        st.write(f"{file_info['filename']}: {status_info['status'].capitalize()}")
                        if status_info["status"] in ["processed", "failed"]:
                            del st.session_state.pending_files[file_id]
                        if status_info["status"] == "error":
                            st.error(f"Error checking status for {file_info['filename']}: {status_info.get('error', 'Unknown error')}")
                    if st.session_state.pending_files:
                        st.button("Manual Refresh", key="refresh_status", on_click=lambda: None)
                        st.write(f"Auto-refreshing in {refresh_interval} seconds...")
                        time.sleep(refresh_interval)
                        st.rerun()  # Updated to st.rerun()
            status_container.empty()

    st.sidebar.markdown("### Uploaded Documents")
    try:
        with st.spinner("Loading documents..."):
            response = requests.get(
                f"{BASE_URL}/documents?user_id={user_id}",
                headers={"X-API-Key": settings.openai_api_key},
                timeout=30
            )
            if response.status_code == 200:
                documents = response.json().get("documents", [])
                st.session_state.file_metadata = documents
                if documents:
                    col1, col2 = st.sidebar.columns([1, 1])
                    with col1:
                        if st.button("Select All"):
                            handle_select_all()
                    with col2:
                        if st.button("Clear All"):
                            handle_clear_all()

                    for file in documents:
                        with st.sidebar.container():
                            cols = st.columns([3, 1, 1])
                            with cols[0]:
                                checkbox_key = f"doc_select_{file['file_id']}"
                                if st.checkbox(
                                    f"{file['filename']} ({file['status']})",
                                    key=checkbox_key,
                                    value=file['file_id'] in st.session_state.selected_docs,
                                    on_change=update_selected_docs
                                ):
                                    pass
                                st.caption(f"{file['file_type']} - {file['upload_date']} - {file['size']/1024:.1f} KB")
                                category = st.selectbox(
                                    "Category",
                                    upload_categories,
                                    index=upload_categories.index(file['category'] or "other"),
                                    key=f"category_{file['file_id']}"
                                )
                                if category != file['category']:
                                    try:
                                        response = requests.patch(
                                            f"{BASE_URL}/documents/{file['file_id']}",
                                            data={"user_id": user_id, "category": category if category != "other" else None},
                                            headers={"X-API-Key": settings.openai_api_key},
                                            timeout=30
                                        )
                                        if response.status_code == 200:
                                            file['category'] = category
                                            st.rerun()
                                        else:
                                            error_msg = f"Failed to update category for {file['filename']}: {response.status_code} - {response.text}"
                                            logger.error(error_msg)
                                            st.error(error_msg)
                                    except Exception as e:
                                        error_msg = f"Error updating category for {file['filename']}: {str(e)}"
                                        logger.error(error_msg)
                                        st.error(error_msg)
                            with cols[1]:
                                preview_url = get_file_preview_url(file['file_id'], user_id)
                                if preview_url:
                                    st.markdown(
                                        f'<a href="{preview_url}" target="_blank" style="text-decoration: none;">🔍</a>',
                                        unsafe_allow_html=True
                                    )
                            with cols[2]:
                                if st.button("🗑️", key=f"delete_{file['file_id']}"):
                                    try:
                                        with st.spinner("Deleting..."):
                                            response = requests.delete(
                                                f"{BASE_URL}/documents/{file['file_id']}?user_id={user_id}",
                                                headers={"X-API-Key": settings.openai_api_key},
                                                timeout=30
                                            )
                                            if response.status_code == 200:
                                                st.session_state.file_metadata = [
                                                    f for f in st.session_state.file_metadata
                                                    if f['file_id'] != file['file_id']
                                                ]
                                                if file['file_id'] in st.session_state.selected_docs:
                                                    st.session_state.selected_docs.remove(file['file_id'])
                                                if file['file_id'] in st.session_state.pending_files:
                                                    del st.session_state.pending_files[file['file_id']]
                                                st.rerun()
                                            else:
                                                error_msg = f"Failed to delete {file['filename']}: {response.status_code} - {response.text}"
                                                logger.error(error_msg)
                                                st.error(error_msg)
                                    except Exception as e:
                                        error_msg = f"Error deleting {file['filename']}: {str(e)}"
                                        logger.error(error_msg)
                                        st.error(error_msg)
                else:
                    st.sidebar.info("No documents uploaded yet.")
            else:
                error_msg = f"Error loading documents: {response.status_code} - {response.text}"
                logger.error(error_msg)
                st.sidebar.error(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error loading documents: {str(e)}"
        logger.error(error_msg)
        st.sidebar.error(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error loading documents: {str(e)}"
        logger.error(error_msg)
        st.sidebar.error(error_msg)

def render_prompt_management(user_id: str):
    st.sidebar.title("Prompt Management")
    with st.sidebar.expander("Create/Edit Prompt"):
        category = st.text_input("Category Name", key="prompt_category")
        prompt_text = st.text_area("Prompt Content", height=200, key="prompt_text")
        if st.button("Save Prompt"):
            if not category or not prompt_text:
                st.error("Category and prompt text are required.")
            else:
                try:
                    response = requests.post(
                        f"{BASE_URL}/prompts",
                        data={"category": category, "prompt": prompt_text, "user_id": user_id},
                        headers={"X-API-Key": settings.openai_api_key},
                        timeout=30
                    )
                    if response.status_code == 200:
                        st.success("Prompt saved successfully")
                        st.session_state.categories = fetch_categories(user_id)
                        st.rerun()
                    else:
                        error_msg = f"Failed to save prompt: {response.status_code} - {response.text}"
                        logger.error(error_msg)
                        st.error(error_msg)
                except Exception as e:
                    error_msg = f"Error saving prompt: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)

    st.sidebar.markdown("### Existing Prompts")
    try:
        response = requests.get(
            f"{BASE_URL}/prompts?user_id={user_id}",
            headers={"X-API-Key": settings.openai_api_key},
            timeout=30
        )
        if response.status_code == 200:
            prompts = response.json().get("prompts", [])
            for prompt in prompts:
                with st.sidebar.expander(f"Prompt: {prompt['category']}"):
                    st.write(f"**Created**: {prompt['created_at']}")
                    st.write(f"**Updated**: {prompt['updated_at']}")
                    st.text_area("Prompt", value=prompt['prompt'], disabled=True, key=f"prompt_view_{prompt['id']}")
                    if st.button("Delete", key=f"delete_prompt_{prompt['id']}"):
                        try:
                            response = requests.delete(
                                f"{BASE_URL}/prompts/{prompt['category']}?user_id={user_id}",
                                headers={"X-API-Key": settings.openai_api_key},
                                timeout=30
                            )
                            if response.status_code == 200:
                                st.success(f"Prompt {prompt['category']} deleted")
                                st.session_state.categories = fetch_categories(user_id)
                                st.rerun()
                            else:
                                error_msg = f"Failed to delete prompt: {response.status_code} - {response.text}"
                                logger.error(error_msg)
                                st.error(error_msg)
                        except Exception as e:
                            error_msg = f"Error deleting prompt: {str(e)}"
                            logger.error(error_msg)
                            st.error(error_msg)
        else:
            st.sidebar.info("No prompts defined yet.")
    except Exception as e:
        error_msg = f"Error loading prompts: {str(e)}"
        logger.error(error_msg)
        st.sidebar.error(error_msg)

def render_chat_sessions(user_id: str):
    st.sidebar.title("Chat Sessions")
    if st.sidebar.button("+ New Chat", key="new_chat_button"):
        create_new_chat()

    try:
        response = requests.get(
            f"{BASE_URL}/chat_sessions?user_id={user_id}",
            headers={"X-API-Key": settings.openai_api_key},
            timeout=30
        )
        if response.status_code == 200:
            sessions = response.json().get("chat_sessions", [])
            st.session_state.chat_sessions = {s["chat_id"]: s for s in sessions}
            for chat_id, chat_data in sorted(
                st.session_state.chat_sessions.items(),
                key=lambda x: x[1]["updated_at"],
                reverse=True
            ):
                if chat_data['user_id'] != user_id:
                    continue
                cols = st.sidebar.columns([4, 1])
                with cols[0]:
                    created_at = datetime.datetime.fromisoformat(chat_data['created_at']).strftime("%Y-%m-%d %H:%M")
                    if st.button(f"Chat {created_at} ({len(chat_data['messages'])} messages)", key=f"chat_{chat_id}"):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.selected_docs = chat_data['document_ids']
                        st.rerun()
                with cols[1]:
                    if st.button("🗑️", key=f"delete_chat_{chat_id}"):
                        try:
                            response = requests.delete(
                                f"{BASE_URL}/chat_sessions/{chat_id}?user_id={user_id}",
                                headers={"X-API-Key": settings.openai_api_key},
                                timeout=30
                            )
                            if response.status_code == 200:
                                if st.session_state.current_chat_id == chat_id:
                                    st.session_state.current_chat_id = None
                                del st.session_state.chat_sessions[chat_id]
                                st.rerun()
                            else:
                                error_msg = f"Failed to delete chat: {response.status_code} - {response.text}"
                                logger.error(error_msg)
                                st.error(error_msg)
                        except Exception as e:
                            error_msg = f"Error deleting chat: {str(e)}"
                            logger.error(error_msg)
                            st.error(error_msg)
        else:
            error_msg = f"Error loading chat sessions: {response.status_code} - {response.text}"
            logger.error(error_msg)
            st.sidebar.error(error_msg)
    except Exception as e:
        error_msg = f"Error loading chat sessions: {str(e)}"
        logger.error(error_msg)
        st.sidebar.error(error_msg)

def render_entity_graph(sources: List[Dict]):
    if not sources:
        return
    net = Network(height="400px", width="100%", directed=True, notebook=True)
    nodes = set()
    node_types = {}

    for source in sources:
        entities = source.get("entities", [])
        relationships = source.get("relationships", [])
        filename = source.get("filename", "Unknown")
        for entity in entities:
            nodes.add(entity)
            node_types[entity] = "entity"
        for rel in relationships:
            subject = rel.get("subject")
            object_ = rel.get("object")
            predicate = rel.get("predicate")
            if subject and object_:
                nodes.add(subject)
                nodes.add(object_)
                node_types[subject] = "entity"
                node_types[object_] = "entity"
                net.add_edge(subject, object_, label=predicate, title=f"From {filename}")

    for node in nodes:
        net.add_node(node, label=node, title=node_types.get(node, "entity"), color="#ADD8E6")

    net.set_options("""
    var options = {
        "nodes": {"font": {"size": 12}},
        "edges": {"font": {"size": 10}, "arrows": "to"},
        "physics": {"barnesHut": {"gravitationalConstant": -2000}}
    }
    """)
    net.save_graph("temp_graph.html")
    with open("temp_graph.html", "r") as f:
        html_content = f.read()
    components.html(html_content, height=400)

def main():
    st.set_page_config(page_title="PDFLLM RAG App", layout="wide", initial_sidebar_state="expanded")
    user_id = "default_user"

    # Fetch categories dynamically
    categories = fetch_categories(user_id)
    query_categories = ["all"] + categories

    render_document_management(user_id)
    render_prompt_management(user_id)
    render_chat_sessions(user_id)

    st.title("PDFLLM RAG App")
    category = st.selectbox("Query Category", query_categories, index=0)

    if st.session_state.current_chat_id:
        chat_data = st.session_state.chat_sessions.get(st.session_state.current_chat_id, {})
        st.markdown(f"### Chat Session: {chat_data.get('created_at', 'New Chat')}")
        chat_container = st.container()
        with chat_container:
            for message in chat_data.get("messages", []):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("Sources"):
                            for source in message["sources"]:
                                st.write(f"**{source['filename']}** (Section {source['chunk_index']}): {source['content'][:200]}...")
                            st.markdown("### Entity-Relationship Graph")
                            render_entity_graph(message["sources"])

    query = st.chat_input("Enter your query")
    if query:
        try:
            with st.spinner("Processing query..."):
                data = {
                    "query": query,
                    "user_id": user_id,
                    "file_ids": st.session_state.selected_docs,
                    "chat_id": st.session_state.current_chat_id,
                    "category": category
                }
                response = requests.post(
                    f"{BASE_URL}/chat",
                    data=data,
                    headers={"X-API-Key": settings.openai_api_key},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.current_chat_id = result["chat_id"]
                    if st.session_state.current_chat_id not in st.session_state.chat_sessions:
                        st.session_state.chat_sessions[st.session_state.current_chat_id] = {
                            "chat_id": result["chat_id"],
                            "user_id": user_id,
                            "created_at": datetime.datetime.now().isoformat(),
                            "updated_at": datetime.datetime.now().isoformat(),
                            "document_ids": st.session_state.selected_docs,
                            "messages": []
                        }
                    st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"].append(
                        {"role": "user", "content": query, "timestamp": datetime.datetime.now().isoformat()}
                    )
                    st.session_state.chat_sessions[st.session_state.current_chat_id]["messages"].append(
                        {
                            "role": "assistant",
                            "content": result["response"],
                            "timestamp": datetime.datetime.now().isoformat(),
                            "sources": result["sources"]
                        }
                    )
                    st.rerun()
                else:
                    error_msg = f"Query failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    st.error(error_msg)
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error processing query: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error processing query: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)

if __name__ == "__main__":
    main()