# OpenNote

An open-source alternative to Google's NotebookLM, designed for intelligent document analysis and interaction with your PDFs using vector search and LLMs.

## 🎯 Project Overview

OpenNote is a Python-based intelligent document analysis and chat system that enables users to interact with their PDF documents using natural language queries. Built as an open-source alternative to Google's NotebookLM, it combines advanced text processing, semantic search, and large language models to provide contextual understanding of your documents.

### Core Architecture

OpenNote implements a **Retrieval-Augmented Generation (RAG)** pipeline with the following components:

- **Document Processing**: Extracts structured text from PDFs using PyMuPDF with smart filtering of headers/footers
- **Vector Database**: Uses ChromaDB with sentence-transformer embeddings for semantic similarity search
- **Intelligent Chunking**: Implements semantic text chunking with overlap and importance scoring
- **LLM Interface**: Supports both OpenAI GPT models and local Ollama models for response generation
- **Conversation Memory**: Maintains context across chat sessions with persistent history

### Technical Stack

- **Python 3.8+**: Core runtime environment
- **ChromaDB**: Vector database for embeddings storage and similarity search
- **Sentence-Transformers**: Text embeddings using models like `all-mpnet-base-v2`
- **PyMuPDF (fitz)**: Advanced PDF text extraction with metadata preservation
- **OpenAI API**: Cloud-based LLM integration for response generation
- **Ollama**: Local LLM support for privacy and cost-efficiency
- **Streamlit**: Optional web interface for document interaction

### ChromaDB Integration Details

The vector store implementation (`vector_store.py:52-514`) provides sophisticated document indexing:

1. **Collection Management**: Creates sanitized collection names following ChromaDB naming conventions
2. **Semantic Chunking**: Splits documents into meaningful chunks (default 1000 chars, 200 overlap) with natural break points
3. **Importance Scoring**: Calculates relevance scores based on content density, numbers, and structural markers
4. **Enhanced Retrieval**: Combines vector similarity with importance weighting for better context selection
5. **Query Enhancement**: Preprocesses queries by removing stopwords and question prefixes

The system stores each chunk with metadata including source document, position, and calculated importance score for intelligent retrieval ranking.

## 🚀 Features

- **Advanced PDF Processing**: Structured text extraction with metadata preservation and header/footer filtering
- **Semantic Search**: ChromaDB-powered vector search with sentence-transformer embeddings
- **Intelligent Text Chunking**: Semantic chunking with importance scoring and natural break points
- **RAG Pipeline**: Context-aware document analysis with Retrieval-Augmented Generation
- **Multiple LLM Providers**: Support for both OpenAI GPT models and local Ollama models
- **Conversation Memory**: Multi-turn interactions with persistent chat history and context management
- **Local-First Architecture**: All data stored locally for privacy and control
- **Notebook Organization**: Organize documents into thematic collections with metadata tracking
- **Enhanced Query Processing**: Query optimization with stopword removal and semantic enhancement
- **Flexible Configuration**: Customizable chunking parameters, retrieval settings, and model options

## 📁 Project Structure

```plaintext
opennote/
├── src/
│   └── opennote/         # Main application code
│       ├── agent.py      # RAG-based AI agent with conversation memory
│       ├── cli.py        # Interactive command-line interface
│       ├── main.py       # Application entry point with argument parsing
│       ├── notebook_manager.py # Notebook creation and organization
│       ├── pdf_processor.py # Advanced PDF text extraction with structure preservation
│       └── vector_store.py # ChromaDB integration with semantic chunking
├── tests/                # Unit tests directory
│   └── test_vector_db.py # Vector database functionality tests
├── examples/             # Example scripts and applications
│   ├── chat_with_notebook.py # Programmatic usage example
│   └── streamlit_app.py  # Web-based UI implementation
├── notebooks/            # User-defined notebook collections
│   └── [notebook_name]/  # Individual notebook directories
│       ├── docs/         # Source PDF documents
│       ├── chromadb/     # ChromaDB vector database files
│       ├── structured_docs/ # Processed document metadata (JSON)
│       ├── history/      # Chat conversation history (JSON)
│       └── metadata.json # Notebook configuration and document tracking
├── configs/              # Configuration files
│   └── config.yaml       # Application configuration
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Build configuration and metadata
└── CLAUDE.md            # Development guidelines and build commands
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

4. Install OpenNote in development mode (enables the `opennote` command):

   ```bash
   pip install -e .
   ```

5. Set up environment variables (create a `.env` file with the following variables):

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-3.5-turbo
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=deepseek-coder:latest
   CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
   DEBUG_MODE=False
   ```

## 📚 Usage

**Important**: OpenNote uses relative imports and must be executed as a module. You can use either:

1. **Command-line tool** (recommended): `opennote` (after installing with `pip install -e .`)
2. **Module execution**: `python -m src.opennote.main`

Do not run the main.py file directly as it will fail due to relative imports.

### Creating a Notebook

```bash
# Using the command-line tool (recommended)
opennote my_notebook --create

# Or using module execution
python -m src.opennote.main my_notebook --create
```

### Processing Documents

Add PDF files to the `notebooks/my_notebook/docs/` directory, then process them:

```bash
# Using the command-line tool
opennote my_notebook --process

# Or using module execution
python -m src.opennote.main my_notebook --process
```

Advanced options for text chunking:

```bash
# Using the command-line tool
opennote my_notebook --process --chunk-size 1500 --chunk-overlap 300

# Or using module execution
python -m src.opennote.main my_notebook --process --chunk-size 1500 --chunk-overlap 300
```

### Chatting with Your Documents

```bash
# Using the command-line tool
opennote my_notebook --chat

# Or using module execution
python -m src.opennote.main my_notebook --chat
```

Using different LLM providers:

```bash
# With OpenAI
opennote my_notebook --chat --provider openai --model gpt-4
python -m src.opennote.main my_notebook --chat --provider openai --model gpt-4

# With Ollama
opennote my_notebook --chat --provider ollama --model llama2
python -m src.opennote.main my_notebook --chat --provider ollama --model llama2
```

Advanced RAG options:

```bash
# Using the command-line tool
opennote my_notebook --chat --top-k 7 --temperature 0.5

# Or using module execution
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

### Intelligent Document Processing

OpenNote's PDF processing pipeline (`pdf_processor.py:35-193`) provides:

- **Structure-Aware Extraction**: Preserves document hierarchy, headings, and formatting
- **Header/Footer Filtering**: Automatically removes page numbers, headers, and footers using position-based detection
- **Metadata Extraction**: Captures document title, author, and table of contents when available
- **Page Break Preservation**: Maintains document structure with clear page demarcations

### Smart Text Chunking

The semantic chunking system (`vector_store.py:142-294`) implements:

- **Section-Based Segmentation**: Identifies logical document sections using paragraph breaks
- **Importance Scoring**: Calculates chunk relevance using content density, numbers, and structural markers
- **Adaptive Sizing**: Respects natural break points while maintaining target chunk sizes
- **Metadata Enrichment**: Each chunk includes position, source, and calculated importance scores

### ChromaDB Population and Querying

The vector database workflow:

1. **Document Ingestion**: PDFs are processed and chunked with importance scoring
2. **Embedding Generation**: Uses sentence-transformers (default: `all-mpnet-base-v2`) for semantic embeddings
3. **Collection Management**: Creates sanitized ChromaDB collections with proper naming conventions
4. **Enhanced Retrieval**: Combines vector similarity with importance weighting for optimal context selection
5. **Query Processing**: Preprocesses queries to remove stopwords and optimize semantic matching

### Conversation Memory System

The agent maintains context through (`agent.py:30-308`):

- **Multi-Turn Context**: Remembers recent conversation exchanges (configurable history length)
- **Persistent Storage**: Saves chat history with timestamps and retrieved context
- **Memory Management**: Commands to save, load, and clear conversation sessions
- **Context Integration**: Incorporates conversation history into response generation prompts

### Flexible LLM Integration

- **OpenAI Support**: GPT-3.5/GPT-4 integration with configurable parameters
- **Ollama Support**: Local model execution for privacy and cost control
- **Temperature Control**: Adjustable response creativity and determinism
- **Top-K Retrieval**: Configurable number of context chunks for response generation

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

### Core Technologies

- **chromadb (0.6.3)**: Vector database for embeddings storage and similarity search
- **sentence-transformers (2.2.2)**: Semantic text embeddings using transformer models
- **PyMuPDF (1.23.26)**: Advanced PDF text extraction with structure preservation
- **python-dotenv (1.0.1)**: Environment variable management and configuration

### LLM Integration

- **openai (1.64.0)**: OpenAI GPT model integration for cloud-based responses
- **requests (2.32.3)**: HTTP client for Ollama local model communication

### Data Processing

- **numpy (2.2.3)**: Numerical computing for embedding operations
- **scikit-learn (1.4.2)**: Machine learning utilities for text processing

### Development Tools

- **black (25.1.0)**: Code formatting with 88-character line limit
- **mypy**: Static type checking for Python code
- **pytest**: Unit testing framework

### Optional UI Components

- **streamlit (1.36.0)**: Web-based user interface for document interaction
- **pandas (2.2.3)**: Data manipulation for UI components

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
