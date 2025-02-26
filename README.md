# OpenNote

An open-source implementation of NotebookLM using Python and ChromaDB for vector storage and retrieval.

## 🎯 Project Overview

OpenNote is an open-source alternative to Google's NotebookLM, designed to provide intelligent document analysis and interaction capabilities. It leverages ChromaDB for efficient vector storage and retrieval, enabling semantic search and contextual understanding of your documents.

## 🚀 Features

- Document vectorisation and storage using ChromaDB
- Semantic search capabilities with intelligent text chunking
- Context-aware document analysis with RAG (Retrieval-Augmented Generation)
- Conversation memory for multi-turn interactions
- Support for multiple LLM providers (OpenAI and Ollama)
=======
- Semantic search capabilities
- Context-aware document analysis
- Local-first architecture for data privacy
- Python-based implementation for easy extensibility

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
│   └── chat_with_notebook.py # Example of programmatic usage
├── notebooks/            # User-defined notebooks
│   └── [notebook_name]/  # Individual notebook directories
│       ├── docs/         # PDF documents
│       ├── chromadb/     # Vector database
│       ├── history/      # Chat history storage
│       └── metadata.json # Notebook metadata
├── .env                  # Environment variables (not in version control)
├── .env.example          # Example environment variables
├── requirements.txt      # Project dependencies
└── pyproject.toml        # Build configuration and metadata
=======
│   └── notebooklm/         # Main application code
│       ├── vector_db.py    # ChromaDB integration and vector operations
│       └── main.py         # Application entry point
├── tests/                  # Unit tests directory
│   └── test_vector_db.py   # Vector database tests
├── configs/                # Configuration files
│   └── config.yaml         # Main configuration
├── data/                   # Local database storage
├── notebooks/              # Jupyter notebooks for experimentation
├── docs/                   # Documentation and project planning
├── .env                    # Environment variables (not in version control)
├── requirements.txt        # Project dependencies
└── pyproject.toml         # Build configuration and metadata
```

## Notebooks Folder Structure

```plaintext
opennote/
│── notebooks/
│   ├── notebook1/
│   │   ├── docs/  (Stores PDFs)
│   │   ├── chromadb/  (Stores vector database)
│   │   ├── metadata.json (Stores metadata about notebook)
│   ├── notebook2/
│   │   ├── docs/
│   │   ├── chromadb/
│   │   ├── metadata.json
│── src/
│   ├── opennote/
│   │   ├── notebook_manager.py  (Handles notebook creation)
│   │   ├── pdf_processor.py  (Extracts text from PDFs)
│   │   ├── vector_store.py  (Handles ChromaDB interactions)
│   ├── main.py
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

4. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 🔧 Configuration

The project uses environment variables for configuration:

- `.env`: Contains API keys and model configurations
  - `OPENAI_API_KEY`: Your OpenAI API key
  - `OPENAI_MODEL`: The OpenAI model to use (default: gpt-3.5-turbo)
  - `OLLAMA_BASE_URL`: URL for Ollama API (default: localhost:11434)
  - `OLLAMA_MODEL`: The Ollama model to use (default: deepseek-coder:latest)
  - `CHROMA_EMBEDDING_MODEL`: The embedding model for ChromaDB (default: all-MiniLM-L6-v2)
  - `DEBUG_MODE`: Enable debug logging (default: False)

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

Using Ollama instead of OpenAI:

```bash
python -m src.opennote.main my_notebook --chat --provider ollama
```

Advanced RAG options:

```bash
python -m src.opennote.main my_notebook --chat --top-k 7 --max-history 15
```

### Saving and Loading Conversations

```bash
# Save chat history at the end of the session
python -m src.opennote.main my_notebook --chat --save-history

# Load a previous conversation
python -m src.opennote.main my_notebook --chat --load-history notebooks/my_notebook/history/chat_history_20240601_123456.json
```

### Using the Example Script

```bash
python examples/chat_with_notebook.py my_notebook --interactive
```

## 🧠 Advanced RAG Features

### Text Chunking

OpenNote implements sophisticated text chunking to improve retrieval quality:

- **Smart Chunking**: Documents are split into smaller, overlapping chunks for more precise retrieval
- **Intelligent Break Points**: The chunking algorithm attempts to find natural break points like paragraphs or sentences
- **Configurable Parameters**:
  - `chunk-size`: Controls the maximum size of each chunk (default: 1000 characters)
  - `chunk-overlap`: Controls the overlap between chunks (default: 200 characters)
- **Metadata Preservation**: Each chunk maintains metadata about its source document and position

### Conversation Memory

OpenNote implements conversation memory for more coherent multi-turn interactions:

- **Context Awareness**: The agent remembers previous exchanges to provide more relevant responses
- **Memory Management**: Commands to save, load, and clear conversation history
- **Configurable Memory Size**: Control how many conversation turns to remember
- **Persistent Storage**: Conversation histories are saved with timestamps in the notebook's history directory
=======
The project uses a combination of `config.yaml` and environment variables for configuration:

- `configs/config.yaml`: General application settings
- `.env`: Sensitive information like API keys

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📚 Documentation

- `docs/`: Contains detailed documentation
- `notebooks/`: Jupyter notebooks with examples and experiments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 Code Style

This project follows PEP 8 guidelines and uses:

- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## 📦 Directory Purpose

### src/opennote/

The main application code resides here:

- `agent.py`: RAG-based AI agent implementation with conversation memory
- `cli.py`: Command-line interface for interactive chat
- `main.py`: Application entry point and command processing
- `notebook_manager.py`: Handles notebook creation and management
- `pdf_processor.py`: Extracts text from PDF documents
- `vector_store.py`: Handles ChromaDB interactions, text chunking, and retrieval

### examples/

Contains example scripts demonstrating how to use OpenNote programmatically:

- `chat_with_notebook.py`: Example of how to use the agent in your own Python code

### notebooks/

Similar to Notebooks in NotebookLM. These are user-defined notebooks that store their own data:

- `docs/`: Stores PDF documents
- `chromadb/`: Stores vector database with chunked text
- `history/`: Stores conversation histories with timestamps
- `metadata.json`: Stores metadata about the notebook and its documents
=======
- `vector_db.py`: Handles all ChromaDB interactions, vector storage, and retrieval operations
- `main.py`: Application entry point and core logic implementation

### tests/

Contains unit tests following pytest conventions:

- `test_vector_db.py`: Tests for vector database operations and integration

### configs/

Configuration management:

- `config.yaml`: Central configuration file for application settings

### data/

Local storage directory:

- ChromaDB files
- Cached vectors
- Temporary processing files

### notebooks/

Similar to Notebooks in NotebookLM. These are user-defined notebooks that store their own pdf data in their own directories.
PDFs are converted to text and then converted to chunks of text. These chunks are then vectorised and stored in the vector database.

### docs/

Project documentation:

- Architecture diagrams
- API documentation
- Development guides

## 🔐 Environment Variables

Required environment variables in `.env`:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-coder:latest
CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEBUG_MODE=False
```

## 📋 Dependencies

Key dependencies (see requirements.txt for complete list):

- chromadb: Vector database for semantic search
- sentence-transformers: Text embedding models
- PyMuPDF: PDF text extraction
- openai: OpenAI API client
- requests: HTTP client for Ollama API
- python-dotenv: Environment variable management
=======
- chromadb: Vector database
- openai: API integration
- python-dotenv: Environment management
- pyyaml: Configuration parsing
- pytest: Testing framework

## 🏗️ Build System

The `pyproject.toml` defines:

- Build system requirements
- Project metadata
- Development dependencies
- Tool configurations

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
