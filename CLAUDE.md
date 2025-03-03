# OpenNote - Guidance for Claude

## Build Commands

- Install: `pip install -r requirements.txt`
- Run app: `python -m src.opennote.main notebook_name --chat/--create/--process`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/test_vector_db.py::test_function_name`
- Format code: `black src/ tests/`
- Type check: `mypy src/`

## Code Style

- **Formatting**: Black with 88 char line limit
- **Imports**: stdlib first, then third-party, then local (use isort)
- **Types**: Use type annotations for all function parameters and returns
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Docstrings for modules, classes, and functions
- **Error handling**: Use specific exceptions, include original error in messages
- **Structure**: Small, focused functions with single responsibilities
- **Testing**: Each function should have corresponding test cases

Maintain existing patterns when modifying files. Follow Black formatting and mypy type checking standards.
