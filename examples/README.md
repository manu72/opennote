# OpenNote Examples

This directory contains example scripts and tools for testing and demonstrating the OpenNote functionality.

## Streamlit Testing UI

The Streamlit app provides a simple graphical interface for testing all aspects of OpenNote:

1. Creating and managing notebooks
2. Uploading and processing PDFs
3. Chatting with your documents using the RAG agent

### Running the Streamlit App

```bash
# Install Streamlit if you haven't already
pip install streamlit

# Run the app
streamlit run examples/streamlit_app.py
```

### Features

- **Notebook Management**: Create new notebooks or select existing ones
- **PDF Processing**: Upload PDFs, configure chunking parameters, and process documents
- **Chat Interface**: Chat with your documents using either OpenAI or Ollama as the LLM provider

## Programmatic Usage Example

The `chat_with_notebook.py` script demonstrates how to use the OpenNote agent programmatically in your own Python code.

### Running the Example Script

```bash
# Basic usage with default settings
python examples/chat_with_notebook.py my_notebook

# Interactive chat mode
python examples/chat_with_notebook.py my_notebook --interactive

# Using Ollama instead of OpenAI
python examples/chat_with_notebook.py my_notebook --provider ollama
```

### Command-line Options

- `notebook_name`: Name of the notebook to chat with (required)
- `--interactive`, `-i`: Start an interactive chat session
- `--provider`: LLM provider to use (`openai` or `ollama`, default: `openai`)
- `--model`: Model name to use (defaults to environment settings)
- `--temperature`: Temperature for response generation (default: 0.7)
- `--top-k`: Number of chunks to retrieve (default: 5)

## Testing Workflow

For a complete testing workflow:

1. Start the Streamlit app: `streamlit run examples/streamlit_app.py`
2. Create a new notebook
3. Upload PDF files
4. Process the PDFs to create the vector database
5. Initialize the agent and start chatting
6. Try different settings and parameters to optimize performance

## Environment Variables

Make sure your `.env` file is properly configured with the necessary API keys and settings:

```
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-coder:latest
CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEBUG_MODE=False
```
