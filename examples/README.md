# OpenNote Examples

This directory contains example scripts and applications demonstrating various ways to use OpenNote for document analysis and RAG-based interactions. These examples showcase both programmatic usage and interactive interfaces for testing and integrating OpenNote into your workflows.

## 📁 Example Applications Overview

### 1. Streamlit Web Interface (`streamlit_app.py`)

A comprehensive web-based testing environment that provides a full-featured UI for exploring OpenNote's capabilities. This application demonstrates:

- **Notebook Management**: Complete CRUD operations for notebook collections
- **Document Processing**: PDF upload, processing, and vector database management  
- **Interactive Chat**: Real-time conversation interface with debug capabilities
- **Configuration Testing**: Dynamic parameter adjustment for optimization

### 2. Programmatic Usage Script (`chat_with_notebook.py`)

A command-line demonstration showing how to integrate OpenNote's agent system into Python applications. Features include:

- **Agent Initialization**: Direct API usage for creating and configuring RAG agents
- **Interactive Mode**: Terminal-based chat interface with session management
- **Batch Processing**: Non-interactive example queries for automated workflows
- **Error Handling**: Robust validation and exception management

## 🌐 Streamlit Web Interface

The Streamlit application (`streamlit_app.py:1-471`) provides a comprehensive testing environment for OpenNote functionality.

### Quick Start

```bash
# Ensure Streamlit is installed (included in requirements.txt)
pip install streamlit

# Launch the web interface
streamlit run examples/streamlit_app.py
```

### Technical Architecture

The Streamlit app implements a multi-tab interface with session state management:

- **Session Persistence**: Maintains notebook selection, agent configuration, and chat history across interactions
- **Real-time Updates**: Dynamic reloading of notebook status and document lists
- **Error Handling**: Comprehensive exception management with user-friendly error messages
- **Custom Styling**: Enhanced UI with custom CSS for improved user experience

### Core Features

#### Notebook Management Tab (`streamlit_app.py:139-232`)
- **Creation**: Real-time notebook initialization with immediate feedback
- **Selection**: Dynamic dropdown populated from existing notebooks  
- **Status Display**: Live metrics showing document count and vector database status
- **Deletion**: Secure notebook removal with confirmation dialog and session cleanup

#### PDF Processing Tab (`streamlit_app.py:234-365`)
- **File Upload**: Multi-file PDF upload with validation and storage to notebook directories
- **Chunking Configuration**: Real-time parameter adjustment for chunk size (100-5000 chars) and overlap (0-1000 chars)
- **Processing Pipeline**: Integration with `process_new_pdfs()` and `store_text_in_chromadb()` functions
- **Document Management**: Individual PDF deletion with instant UI updates
- **Vector Database Control**: Complete regeneration capability with metadata reset

#### Chat Interface Tab (`streamlit_app.py:367-469`)
- **Agent Configuration**: Dynamic LLM provider selection (OpenAI/Ollama) with model specification
- **Parameter Control**: Temperature (0.0-1.0) and top-k retrieval (1-20) adjustment
- **Real-time Chat**: Streaming conversation interface with message persistence
- **Debug Information**: Expandable view of retrieved document chunks with relevance scores
- **API Status**: Environment variable validation and configuration feedback

## 💻 Programmatic Usage Script

The command-line script (`chat_with_notebook.py:1-126`) demonstrates direct integration of OpenNote's agent system into Python applications.

### Technical Implementation

The script showcases key integration patterns:

- **Path Management**: Dynamic sys.path modification for package imports (`chat_with_notebook.py:18-19`)
- **Validation Pipeline**: Pre-flight checks for notebook existence and vector database availability (`chat_with_notebook.py:37-48`)
- **Agent Factory**: Direct usage of `create_agent()` function with parameter passing (`chat_with_notebook.py:54-60`)
- **Session Management**: Interactive mode with command processing and history operations (`chat_with_notebook.py:72-106`)

### Usage Patterns

#### Non-Interactive Mode (Batch Processing)
```bash
# Automated query processing with predefined questions
python examples/chat_with_notebook.py research_notebook --model gpt-4o
```
Executes three example queries (`chat_with_notebook.py:111-121`) demonstrating automated document analysis workflows.

#### Interactive Mode (Session-Based)
```bash
# Terminal-based chat interface with session persistence
python examples/chat_with_notebook.py research_notebook --interactive --model gpt-4o
```
Provides full interactive capabilities with commands:
- `clear`: Reset conversation memory
- `save`: Persist chat history to JSON
- `help`: Display available commands
- `exit/quit/q`: Graceful session termination

#### Advanced Configuration
```bash
# Fine-tuned retrieval and generation parameters
python examples/chat_with_notebook.py research_notebook \
  --provider ollama \
  --model deepseek-coder:latest \
  --temperature 0.3 \
  --top-k 7 \
  --interactive
```

### Command-Line Interface

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `notebook_name` | str | required | Target notebook identifier for agent initialization |
| `--interactive` | flag | false | Enables terminal-based interactive chat mode |
| `--provider` | choice | openai | LLM provider selection (`openai` or `ollama`) |
| `--model` | str | required | Specific model name for response generation |
| `--temperature` | float | 0.7 | Response creativity control (0.0-1.0) |
| `--top-k` | int | 5 | Number of context chunks for retrieval (1-20) |

### Integration Examples

The script demonstrates several integration patterns for custom applications:

1. **Agent Initialization** (`chat_with_notebook.py:54-61`):
```python
agent = create_agent(
    notebook_name=notebook_name,
    llm_provider=provider,
    model_name=model,
    temperature=temperature,
    top_k=top_k
)
```

2. **Query Processing** (`chat_with_notebook.py:101-102`):
```python
response = agent.chat(user_input)
```

3. **History Management** (`chat_with_notebook.py:81-86`):
```python
agent.clear_history()
history_path = agent.save_chat_history()
```

## 🧪 Comprehensive Testing Workflow

### End-to-End Testing Process

1. **Environment Setup**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   
   # Configure environment variables in .env file
   cp .env.example .env  # Edit with your API keys
   ```

2. **Web Interface Testing**
   ```bash
   # Launch Streamlit application
   streamlit run examples/streamlit_app.py
   ```
   - Create notebook → Upload PDFs → Configure chunking → Process documents → Initialize agent → Chat

3. **Programmatic Testing**
   ```bash
   # Test batch processing mode
   python examples/chat_with_notebook.py test_notebook --model gpt-4o
   
   # Test interactive mode with custom parameters
   python examples/chat_with_notebook.py test_notebook --interactive \
     --provider ollama --model llama2 --temperature 0.3 --top-k 7
   ```

4. **Parameter Optimization Testing**
   - **Chunking**: Test different chunk sizes (500-2000) and overlaps (100-400) for various document types
   - **Retrieval**: Experiment with top-k values (3-10) to balance context breadth vs. precision
   - **Generation**: Adjust temperature (0.1-0.9) based on use case (factual vs. creative responses)

### Performance Benchmarking

Use the examples to test system performance with various configurations:

- **Document Size**: Test with different PDF sizes (1-100 pages)
- **Collection Scale**: Evaluate performance with 1-100 documents per notebook
- **Query Complexity**: Test simple factual queries vs. complex analytical questions
- **Provider Comparison**: Compare OpenAI vs. Ollama response quality and speed

## ⚙️ Configuration Management

### Required Environment Variables

```bash
# Essential for OpenAI integration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o  # or gpt-3.5-turbo, gpt-4

# Optional for local Ollama integration  
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-coder:latest  # or llama2, mistral

# ChromaDB embedding configuration
CHROMA_EMBEDDING_MODEL=all-mpnet-base-v2  # or all-MiniLM-L6-v2

# Development settings
DEBUG_MODE=False
```

### Configuration Validation

Both examples include validation for:
- **API Key Presence**: Verification of required environment variables
- **Notebook Existence**: Pre-flight checks for target notebook availability  
- **Vector Database Status**: Confirmation of processed documents and ChromaDB collections
- **Model Accessibility**: Provider-specific connectivity validation

## 🔧 Troubleshooting Examples

### Common Issues and Solutions

1. **"Notebook does not exist"**
   - Use main script to create: `python -m src.opennote.main notebook_name --create`
   - Or use Streamlit interface notebook management tab

2. **"Vector database not created"**
   - Process PDFs first: `python -m src.opennote.main notebook_name --process`
   - Or use Streamlit PDF processing tab with chunking configuration

3. **"Model name must be provided"**
   - Specify model explicitly: `--model gpt-4o` or `--model llama2`
   - Set default in environment variables

4. **OpenAI API errors**
   - Verify API key in `.env` file
   - Check API quota and billing status
   - Test with different model names

5. **Ollama connection issues**
   - Ensure Ollama is running: `ollama serve`
   - Verify model is available: `ollama list`
   - Check base URL configuration
