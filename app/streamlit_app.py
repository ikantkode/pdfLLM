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
FASTAPI_HOST = "rag-service"
FASTAPI_PORT = "8000"

class SessionState:
    def __init__(self):
        self._initialize_state()

    def _initialize_state(self):
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

session_state = SessionState()

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
    return f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/preview/{file_id}?user_id={user_id}&X-API-Key={settings.openai_api_key}"

def render_document_management(user_id: str):
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
        category = st.selectbox("Category", ["submittals", "payrolls", "bank_statements", "other"], index=0)

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
                    data = {"user_id": user_id, "category": category if category != "other" else None}
                    try:
                        response = requests.post(
                            f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/process_file",
                            files=files,
                            data=data,
                            headers={"X-API-Key": settings.openai_api_key}
                        )
                        if response.status_code == 200:
                            success_count += 1
                        else:
                            error_messages.append(f"Upload failed for {uploaded_file.name}")
                    except Exception as e:
                        error_messages.append(f"Upload error: {str(e)}")

            if success_count > 0:
                st.success(f"Uploaded {success_count} file(s)")
                st.session_state.upload_key += 1
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
                                ["submittals", "payrolls", "bank_statements", "other"],
                                index=["submittals", "payrolls", "bank_statements", "other"].index(file['category'] or "other"),
                                key=f"category_{file['file_id']}"
                            )
                            if category != file['category']:
                                try:
                                    response = requests.patch(
                                        f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/documents/{file['file_id']}",
                                        data={"user_id": user_id, "category": category if category != "other" else None},
                                        headers={"X-API-Key": settings.openai_api_key}
                                    )
                                    if response.status_code == 200:
                                        file['category'] = category
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to update category: {response.text}")
                                except Exception as e:
                                    st.error(f"Error updating category: {str(e)}")
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
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Delete error: {str(e)}")
            else:
                st.sidebar.info("No documents uploaded yet.")
        else:
            st.sidebar.error(f"Error loading documents: {response.text}")
    except Exception as e:
        st.sidebar.error(f"Error loading documents: {str(e)}")

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
                        f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/prompts",
                        data={"category": category, "prompt": prompt_text, "user_id": user_id},
                        headers={"X-API-Key": settings.openai_api_key}
                    )
                    if response.status_code == 200:
                        st.success("Prompt saved successfully")
                        st.rerun()
                    else:
                        st.error(f"Failed to save prompt: {response.text}")
                except Exception as e:
                    st.error(f"Error saving prompt: {str(e)}")

    st.sidebar.markdown("### Existing Prompts")
    try:
        response = requests.get(
            f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/prompts?user_id={user_id}",
            headers={"X-API-Key": settings.openai_api_key}
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
                                f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/prompts/{prompt['category']}?user_id={user_id}",
                                headers={"X-API-Key": settings.openai_api_key}
                            )
                            if response.status_code == 200:
                                st.success(f"Prompt {prompt['category']} deleted")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete prompt: {response.text}")
                        except Exception as e:
                            st.error(f"Error deleting prompt: {str(e)}")
        else:
            st.sidebar.info("No prompts defined yet.")
    except Exception as e:
        st.sidebar.error(f"Error loading prompts: {str(e)}")

def render_chat_sessions(user_id: str):
    st.sidebar.title("Chat Sessions")
    if st.sidebar.button("+ New Chat", key="new_chat_button"):
        create_new_chat()

    try:
        response = requests.get(
            f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/chat_sessions?user_id={user_id}",
            headers={"X-API-Key": settings.openai_api_key}
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
                                f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/chat_sessions/{chat_id}?user_id={user_id}",
                                headers={"X-API-Key": settings.openai_api_key}
                            )
                            if response.status_code == 200:
                                if st.session_state.current_chat_id == chat_id:
                                    st.session_state.current_chat_id = None
                                del st.session_state.chat_sessions[chat_id]
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting chat: {str(e)}")
        else:
            st.sidebar.error(f"Error loading chat sessions: {response.text}")
    except Exception as e:
        st.sidebar.error(f"Error loading chat sessions: {str(e)}")

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
            source_node = rel.get("source")
            target_node = rel.get("target")
            relation = rel.get("relation")
            if source_node and target_node:
                nodes.add(source_node)
                nodes.add(target_node)
                node_types[source_node] = "entity"
                node_types[target_node] = "entity"
                net.add_edge(source_node, target_node, label=relation, title=f"From {filename}")

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

    render_document_management(user_id)
    render_prompt_management(user_id)
    render_chat_sessions(user_id)

    st.title("PDFLLM RAG App")
    category = st.selectbox("Query Category", ["all", "submittals", "payrolls", "bank_statements"], index=0)

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
                    f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/chat",
                    data=data,
                    headers={"X-API-Key": settings.openai_api_key}
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
                    st.error(f"Query failed: {response.text}")
        except Exception as e:
            st.error(f"Query error: {str(e)}")

if __name__ == "__main__":
    main()