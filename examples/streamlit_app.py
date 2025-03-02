"""
Streamlit app for testing OpenNote functionality.

This app provides a simple UI for:
1. Creating new notebooks
2. Adding PDFs to notebooks
3. Processing PDFs and confirming ChromaDB creation
4. Chatting with the notebook using the RAG agent

Run with: streamlit run examples/streamlit_app.py
"""
import os
import sys
import time
import json
import shutil
import streamlit as st  # type: ignore
from pathlib import Path

# Add the project root to the path so we can import the opennote package
sys.path.append(str(Path(__file__).parent.parent))

from src.opennote.notebook_manager import create_notebook
from src.opennote.pdf_processor import process_new_pdfs
from src.opennote.vector_store import store_text_in_chromadb, query_vector_db
from src.opennote.agent import create_agent

# Set page configuration
st.set_page_config(
    page_title="OpenNote Testing UI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = None

if "agent" not in st.session_state:
    st.session_state.agent = None

def list_notebooks():
    """List all available notebooks."""
    notebooks_dir = Path("notebooks")
    if not notebooks_dir.exists():
        return []

    return [d.name for d in notebooks_dir.iterdir() if d.is_dir() and (d / "metadata.json").exists()]

def upload_and_save_pdfs(notebook_name, uploaded_files):
    """Save uploaded PDF files to the notebook's docs directory."""
    if not uploaded_files:
        return []

    docs_dir = Path(f"notebooks/{notebook_name}/docs")
    docs_dir.mkdir(exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = docs_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(uploaded_file.name)

    return saved_files

def main():
    """Main function for the Streamlit app."""
    st.title("OpenNote Testing UI")
    st.markdown("A simple interface for testing OpenNote functionality")
    
    # Add help section with expandable container
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Getting Started with OpenNote
        
        This app allows you to test the OpenNote RAG (Retrieval-Augmented Generation) system with your own documents. Follow these steps:
        
        1. **Create or Manage Notebooks**: Start by creating a new notebook or selecting an existing one in the Notebook Management tab. You can also delete unwanted notebooks in the "Danger Zone" section.
        
        2. **Upload Documents**: In the PDF Processing tab, upload your PDF documents and save them to your notebook. You can also delete PDFs using the trash icon next to each file.
        
        3. **Process Documents**: Configure chunking parameters and process your PDFs to create the vector database. If you need to regenerate the vector database with new settings, use the "Regenerate Vector Database" button.
        
        4. **Chat with Your Documents**: In the Chat Interface tab, configure your LLM settings, initialize the agent, and start asking questions about your documents.
        
        ### Understanding Configuration Options
        
        - **Chunk Size**: Controls how large each piece of text is when splitting your documents. Larger chunks provide more context but may reduce precision.
        
        - **Chunk Overlap**: Determines how much text overlaps between consecutive chunks to maintain context across chunk boundaries.
        
        - **LLM Provider**: Choose between OpenAI (cloud-based) or Ollama (local) models.
        
        - **Temperature**: Controls the randomness in responses. Lower values (0.0-0.3) give more focused answers, while higher values (0.7-1.0) produce more creative outputs.
        
        - **Top K Chunks**: The number of document chunks to retrieve for each query. More chunks provide broader context but may include less relevant information.
        
        ### Tips for Best Results
        
        - Use descriptive notebook names to organize your documents by topic or project.
        - For longer documents, consider using larger chunk sizes (1000-1500) with more overlap (200-300).
        - Start with a lower temperature (0.3-0.5) for factual questions and higher (0.7-0.9) for creative tasks.
        - Experiment with different Top K values to find the right balance between focused and comprehensive responses.
        """)
    
    # Create tabs for different functionality
    tabs = st.tabs(["Notebook Management", "PDF Processing", "Chat Interface"])

    # Tab 1: Notebook Management
    with tabs[0]:
        st.header("Notebook Management")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Create New Notebook")
            new_notebook_name = st.text_input("Notebook Name", key="new_notebook")
            create_button = st.button("Create Notebook")

            if create_button and new_notebook_name:
                try:
                    notebook_path = create_notebook(new_notebook_name)
                    st.success(f"Notebook '{new_notebook_name}' created successfully at {notebook_path}")
                    st.session_state.current_notebook = new_notebook_name
                except Exception as e:
                    st.error(f"Error creating notebook: {str(e)}")

        with col2:
            st.subheader("Select Existing Notebook")
            notebooks = list_notebooks()
            if notebooks:
                selected_notebook = st.selectbox(
                    "Choose a notebook", 
                    notebooks,
                    index=notebooks.index(st.session_state.current_notebook) if st.session_state.current_notebook in notebooks else 0
                )
                if st.button("Select Notebook"):
                    st.session_state.current_notebook = selected_notebook
                    st.success(f"Notebook '{selected_notebook}' selected")
            else:
                st.info("No notebooks found. Create a new one.")

        # Display current notebook info
        st.divider()
        if st.session_state.current_notebook:
            st.subheader("Current Notebook")
            notebook_path = Path(f"notebooks/{st.session_state.current_notebook}")

            # Get notebook stats
            docs_path = notebook_path / "docs"
            chroma_path = notebook_path / "chromadb"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Notebook", st.session_state.current_notebook)

            with col2:
                pdf_count = len(list(docs_path.glob("*.pdf"))) if docs_path.exists() else 0
                st.metric("PDF Documents", pdf_count)

            with col3:
                has_vectors = chroma_path.exists() and any(chroma_path.iterdir())
                st.metric("Vector DB", "Created" if has_vectors else "Not created")

            st.text(f"Notebook path: {notebook_path.absolute()}")
            st.text(f"ChromaDB path: {chroma_path.absolute()}")
            
            # Add notebook deletion section
            st.divider()
            st.subheader("Danger Zone")
            
            delete_col1, delete_col2 = st.columns([3, 1])
            
            with delete_col1:
                st.warning("Deleting a notebook will permanently remove all associated PDFs, vectors, and data.")
            
            with delete_col2:
                # Use a delete button with confirmation
                if st.button("Delete Notebook", type="primary", help="Permanently delete this notebook and all its data"):
                    # Create a confirmation dialog
                    delete_confirmation = st.text_input(
                        f"Type '{st.session_state.current_notebook}' to confirm deletion:",
                        key="delete_confirmation"
                    )
                    
                    if delete_confirmation == st.session_state.current_notebook:
                        try:
                            # Delete the entire notebook directory
                            shutil.rmtree(notebook_path)
                            
                            # Reset session state
                            st.session_state.current_notebook = None
                            st.session_state.agent = None
                            st.session_state.messages = []
                            
                            st.success(f"Notebook '{delete_confirmation}' has been permanently deleted.")
                            time.sleep(2)  # Give user time to see the message
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting notebook: {str(e)}")
                    elif delete_confirmation:
                        st.error("Notebook name doesn't match. Deletion cancelled.")

    # Tab 2: PDF Processing
    with tabs[1]:
        st.header("PDF Processing")

        if not st.session_state.current_notebook:
            st.warning("Please select or create a notebook first")
        else:
            st.subheader(f"Upload PDFs to '{st.session_state.current_notebook}'")

            uploaded_files = st.file_uploader(
                "Upload PDF files", 
                type="pdf", 
                accept_multiple_files=True,
                key="pdf_uploader"
            )

            col1, col2 = st.columns([1, 1])

            with col1:
                if st.button("Save PDFs") and uploaded_files:
                    saved_files = upload_and_save_pdfs(st.session_state.current_notebook, uploaded_files)
                    if saved_files:
                        st.success(f"Saved {len(saved_files)} PDF files: {', '.join(saved_files)}")
                    else:
                        st.info("No files were uploaded")

            with col2:
                chunk_size = st.number_input(
                    "Chunk Size", 
                    min_value=100, 
                    max_value=5000, 
                    value=1000,
                    help="The maximum size of each text chunk in characters. Larger chunks provide more context but may reduce precision. Recommended range: 500-2000."
                )
                chunk_overlap = st.number_input(
                    "Chunk Overlap", 
                    min_value=0, 
                    max_value=1000, 
                    value=200,
                    help="The number of characters that overlap between consecutive chunks. Higher overlap helps maintain context across chunks. Typically 10-20% of chunk size."
                )

                if st.button("Process PDFs"):
                    with st.spinner("Processing PDFs..."):
                        try:
                            # Process new PDFs
                            extracted_texts = process_new_pdfs(st.session_state.current_notebook)

                            if extracted_texts:
                                # Store in ChromaDB
                                store_text_in_chromadb(
                                    st.session_state.current_notebook,
                                    extracted_texts,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap
                                )
                                st.success(f"Processed and stored {len(extracted_texts)} documents in ChromaDB")
                            else:
                                st.info("No new documents to process")
                        except Exception as e:
                            st.error(f"Error processing PDFs: {str(e)}")

            # Display current PDFs with delete buttons
            st.divider()
            st.subheader("Current PDFs")

            docs_path = Path(f"notebooks/{st.session_state.current_notebook}/docs")
            if docs_path.exists():
                pdfs = list(docs_path.glob("*.pdf"))
                if pdfs:
                    pdf_names = [pdf.name for pdf in pdfs]
                    st.write(f"Found {len(pdf_names)} PDFs:")
                    
                    # Display PDFs with delete buttons
                    for i, pdf in enumerate(pdf_names):
                        col1, col2 = st.columns([6, 1])
                        with col1:
                            st.text(f"• {pdf}")
                        with col2:
                            delete_key = f"delete_{i}_{pdf}"
                            if st.button("🗑️", key=delete_key, help=f"Delete {pdf}"):
                                try:
                                    # Delete the PDF file
                                    file_to_delete = docs_path / pdf
                                    if file_to_delete.exists():
                                        file_to_delete.unlink()
                                        st.success(f"Deleted {pdf}")
                                        # Reload the page to update the list
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting file: {str(e)}")
                else:
                    st.info("No PDFs found in this notebook")
            else:
                st.info("Documents directory does not exist yet")
                
            # Add option to regenerate the vector database
            st.divider()
            st.subheader("Vector Database Management")
            
            chroma_path = Path(f"notebooks/{st.session_state.current_notebook}/chromadb")
            has_vectors = chroma_path.exists() and any(chroma_path.iterdir())
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.metric("Vector DB Status", "Created" if has_vectors else "Not created")
            
            with col2:
                if has_vectors:
                    if st.button("Regenerate Vector Database", 
                                  help="This will delete the existing vector database and recreate it from all PDFs using current chunking settings"):
                        try:
                            # Delete existing ChromaDB directory
                            shutil.rmtree(chroma_path)
                            st.success("Existing vector database deleted.")
                            
                            # Update notebook metadata to reprocess all PDFs
                            metadata_path = Path(f"notebooks/{st.session_state.current_notebook}/metadata.json")
                            if metadata_path.exists():
                                with open(metadata_path, "r") as f:
                                    metadata = json.load(f)
                                
                                # Clear processed documents list to force reprocessing
                                metadata["documents"] = []
                                
                                with open(metadata_path, "w") as f:
                                    json.dump(metadata, f, indent=4)
                                
                                st.info("Ready to regenerate. Please click 'Process PDFs' to rebuild the vector database with current settings.")
                        except Exception as e:
                            st.error(f"Error regenerating vector database: {str(e)}")

    # Tab 3: Chat Interface
    with tabs[2]:
        st.header("Chat with Your Notebook")

        if not st.session_state.current_notebook:
            st.warning("Please select or create a notebook first")
        else:
            # Check if the notebook has been processed
            chroma_path = Path(f"notebooks/{st.session_state.current_notebook}/chromadb")
            if not chroma_path.exists() or not any(chroma_path.iterdir()):
                st.warning("This notebook doesn't have a vector database yet. Please process PDFs first.")
            else:
                # Chat settings
                with st.sidebar:
                    st.header("Chat Settings")
                    
                    # Display API key status
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    if openai_api_key:
                        st.success("✅ OPENAI_API_KEY is configured")
                    else:
                        st.error("❌ OPENAI_API_KEY is not set. Please add it to your .env file.")
                    
                    provider = st.selectbox(
                        "LLM Provider", 
                        ["openai", "ollama"], 
                        index=0,
                        help="Select the AI provider to use. OpenAI offers cloud-based models like GPT-4, while Ollama provides locally-run open-source models."
                    )
                    model = st.text_input(
                        "Model Name", 
                        value="gpt-4o" if provider == "openai" else "deepseek-coder:latest",
                        help="The specific model to use. For OpenAI, try gpt-3.5-turbo, gpt-4, or gpt-4o. For Ollama, try llama2, mistral, or deepseek-coder."
                    )
                    temperature = st.slider(
                        "Temperature", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.7, 
                        step=0.1,
                        help="Controls randomness in responses. Lower values (0.0-0.3) give more focused, deterministic responses. Higher values (0.7-1.0) produce more creative, varied outputs."
                    )
                    top_k = st.slider(
                        "Top K Chunks", 
                        min_value=1, 
                        max_value=20, 
                        value=5,
                        help="The number of most relevant document chunks to retrieve for each query. More chunks provide broader context but may include less relevant information."
                    )

                    if st.button("Initialize Agent"):
                        with st.spinner("Initializing agent..."):
                            try:
                                st.session_state.agent = create_agent(
                                    notebook_name=st.session_state.current_notebook,
                                    llm_provider=provider,
                                    model_name=model,
                                    temperature=temperature,
                                    top_k=top_k
                                )
                                st.success("Agent initialized successfully")
                            except Exception as e:
                                st.error(f"Error initializing agent: {str(e)}")

                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Ask a question about your documents..."):
                    if not st.session_state.agent:
                        st.warning("Please initialize the agent first")
                    else:
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})

                        # Display user message
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        # Generate response
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                try:
                                    response = st.session_state.agent.chat(prompt)
                                    st.markdown(response)

                                    # Add assistant response to chat history
                                    st.session_state.messages.append({"role": "assistant", "content": response})
                                    
                                    # Display retrieved chunks if available
                                    if hasattr(st.session_state.agent, 'last_retrieved_chunks') and st.session_state.agent.last_retrieved_chunks:
                                        with st.expander("View retrieved document chunks"):
                                            for i, chunk in enumerate(st.session_state.agent.last_retrieved_chunks):
                                                st.markdown(f"**Chunk {i+1}** (Source: {chunk.get('metadata', {}).get('source', 'Unknown')})")
                                                st.markdown(f"```\n{chunk.get('text', '')[:300]}...\n```")
                                                st.markdown(f"Relevance Score: {chunk.get('relevance_score', 'N/A'):.2f}" if chunk.get('relevance_score') else "Relevance Score: N/A")
                                                st.divider()
                                except Exception as e:
                                    error_msg = f"Error generating response: {str(e)}"
                                    st.error(error_msg)
                                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main() 