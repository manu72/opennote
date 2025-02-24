# OpenNote

An open-source implementation of NotebookLM using Python and ChromaDB for vector storage and retrieval.

## 🎯 Project Overview

OpenNote is an open-source alternative to Google's NotebookLM, designed to provide intelligent document analysis and interaction capabilities. It leverages ChromaDB for efficient vector storage and retrieval, enabling semantic search and contextual understanding of your documents.

## 🚀 Features

- Document vectorisation and storage using ChromaDB
- Semantic search capabilities
- Context-aware document analysis
- Local-first architecture for data privacy
- Python-based implementation for easy extensibility

## 📁 Project Structure

```plaintext
opennote/
├── src/
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
OPENAI_API_KEY=your_api_key_here
CHROMA_DB_PATH=./data/chromadb
DEBUG_MODE=False
```

## 📋 Dependencies

Key dependencies (see requirements.txt for complete list):

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
