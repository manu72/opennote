# OpenNote

An open-source alternative to Google's NotebookLM, designed for intelligent document analysis and interaction with your PDFs using vector search and LLMs.

## 🎯 Project Overview

OpenNote is an open-source alternative to Google's NotebookLM, designed to provide intelligent document analysis and interaction capabilities. It leverages ChromaDB for efficient vector storage and retrieval, enabling semantic search and contextual understanding of your documents.

## 🚀 Features

- **Document Processing**: Extract and process text from PDF documents
- **Semantic Search**: Vector-based search capabilities with intelligent text chunking
- **RAG Integration**: Context-aware document analysis with Retrieval-Augmented Generation
- **Multiple LLM Providers**: Support for both OpenAI and Ollama
- **Conversation Memory**: Multi-turn interactions with history management
- **Local-First Architecture**: All data stored locally for privacy
- **Notebook Organization**: Organize documents into thematic notebooks

## 📁 Project Structure

```plaintext
opennote/
├── src/
│   └── opennote/         # Main application code
│       ├── agent.py      # RAG-based AI agent implementation
│       ├── cli.py        # Command-line interface
│       ├── main.py       # Application entry point
│       ├── notebook_manager.py # Notebook creation and management
│       ├── pdf_processor.py # PDF text extraction
│       └── vector_store.py # ChromaDB integration and vector operations
├── tests/                # Unit tests directory
├── examples/             # Example scripts for using OpenNote
│   ├── chat_with_notebook.py # Example of programmatic usage
│   └── streamlit_app.py  # Web-based UI implementation
├── notebooks/            # User-defined notebooks
│   └── [notebook_name]/  # Individual notebook directories
│       ├── docs/         # PDF documents
│       ├── chromadb/     # Vector database
│       ├── history/      # Chat history storage
│       └── metadata.json # Notebook metadata
├── requirements.txt      # Project dependencies
└── pyproject.toml        # Build configuration and metadata
```

## 🛠️ Setup and Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/opennote.git
   cd opennote
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file with the following variables):

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=deepseek-coder:latest
   CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
   DEBUG_MODE=False
   ```

## 📚 Usage

### Creating a Notebook

```bash
python -m src.opennote.main my_notebook --create
```

### Processing Documents

Add PDF files to the `notebooks/my_notebook/docs/` directory, then process them:

```bash
python -m src.opennote.main my_notebook --process
```

Advanced options for text chunking:

```bash
python -m src.opennote.main my_notebook --process --chunk-size 1500 --chunk-overlap 300
```

### Chatting with Your Documents

```bash
python -m src.opennote.main my_notebook --chat
```

Using different LLM providers:

```bash
# With OpenAI
python -m src.opennote.main my_notebook --chat --provider openai --model gpt-4

# With Ollama
python -m src.opennote.main my_notebook --chat --provider ollama --model llama2
```

Advanced RAG options:

```bash
python -m src.opennote.main my_notebook --chat --top-k 7 --temperature 0.5
```

### Saving and Loading Conversations

```bash
# Save chat history at the end of the session
python -m src.opennote.main my_notebook --chat --save-history

# Load a previous conversation
python -m src.opennote.main my_notebook --chat --load-history notebooks/my_notebook/history/chat_history_20240601_123456.json
```

### Using the Streamlit UI

```bash
streamlit run examples/streamlit_app.py
```

## 🧠 Advanced Features

### Smart Text Chunking

OpenNote implements sophisticated text chunking to improve retrieval quality:

- **Semantic Chunking**: Documents are split into smaller, overlapping chunks for more precise retrieval
- **Intelligent Break Points**: The chunking algorithm attempts to find natural break points like paragraphs or sentences
- **Importance Scoring**: Information-dense chunks are prioritized during retrieval
- **Metadata Preservation**: Each chunk maintains metadata about its source document and position

### Conversation Memory

OpenNote implements conversation memory for more coherent multi-turn interactions:

- **Context Awareness**: The agent remembers previous exchanges to provide more relevant responses
- **Memory Management**: Commands to save, load, and clear conversation history
- **Configurable Memory Size**: Control how many conversation turns to remember
- **Persistent Storage**: Conversation histories are saved with timestamps

### Multiple LLM Providers

- **OpenAI Integration**: Use powerful cloud-based models
- **Ollama Integration**: Use local open-source models for privacy and cost-savings
- **Configurable Parameters**: Control temperature, top-k retrieval, and more

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run a specific test:

```bash
pytest tests/test_vector_db.py::test_function_name
```

## 📝 Code Style

This project follows Black formatting standards with:

- **Formatting**: Black with 88 char line limit
- **Imports**: stdlib first, then third-party, then local (use isort)
- **Types**: Mypy for type checking
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Docstrings for modules, classes, and functions

Format code with:

```bash
black src/ tests/
```

Type check with:

```bash
mypy src/
```

## 📦 Key Dependencies

- **chromadb**: Vector database for semantic search
- **sentence-transformers**: Text embedding models
- **PyMuPDF**: PDF text extraction
- **openai**: OpenAI API client
- **streamlit**: Web UI framework
- **python-dotenv**: Environment variable management

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.