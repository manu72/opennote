"""
Script creates a notebook, adds PDFs, processes them, and generates the ChromaDB vector store.
It can also start an interactive chat with the AI agent.
"""
import os
import argparse
from .notebook_manager import create_notebook
from .pdf_processor import process_new_pdfs
from .vector_store import store_text_in_chromadb
from .agent import create_agent
from .cli import chat_loop

def main() -> None:
    """
    Main function to create a notebook, process new PDFs, store the text in ChromaDB,
    and optionally start an interactive chat with the AI agent.
    """
    parser = argparse.ArgumentParser(description="OpenNote - NotebookLM alternative")
    parser.add_argument("notebook_name", help="Name of the notebook to use")
    parser.add_argument("--process", "-p", action="store_true",
                        help="Process new PDFs in the notebook")
    parser.add_argument("--chat", "-c", action="store_true",
                        help="Start an interactive chat with the AI agent")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                        help="LLM provider to use for chat (default: openai)")
    parser.add_argument("--model", help="Model name to use for chat")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for response generation (default: 0.7)")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--max-history", type=int, default=10,
                        help="Maximum conversation turns to remember (default: 10)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Size of text chunks in characters (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                        help="Overlap between chunks in characters (default: 200)")
    parser.add_argument("--save-history", "-s", action="store_true",
                        help="Save chat history at the end of the session")
    parser.add_argument("--load-history", "-l",
                        help="Path to load chat history from")
    parser.add_argument("--create", action="store_true",
                        help="Create the notebook if it doesn't exist")

    args = parser.parse_args()

    # Validate arguments
    if args.chunk_overlap >= args.chunk_size:
        print("Error: chunk-overlap must be less than chunk-size")
        return

    if args.temperature < 0 or args.temperature > 2:
        print("Error: temperature must be between 0 and 2")
        return

    if args.top_k <= 0:
        print("Error: top-k must be positive")
        return

    # Create or check the notebook
    if args.create:
        create_notebook(args.notebook_name)
    else:
        notebook_path = os.path.join("notebooks", args.notebook_name)
        if not os.path.exists(notebook_path):
            print(f"Error: Notebook '{args.notebook_name}' does not exist.")
            print("Use --create flag to create it, or run:")
            print(f"  opennote {args.notebook_name} --create")
            print(f"  # or")
            print(f"  python -m src.opennote.main {args.notebook_name} --create")
            return
        create_notebook(args.notebook_name)  # Ensure subdirectories exist

    # Process new PDFs if requested
    if args.process:
        extracted_texts = process_new_pdfs(args.notebook_name)
        if extracted_texts:
            store_text_in_chromadb(
                args.notebook_name,
                extracted_texts,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            print(f"Processed and stored {len(extracted_texts)} new documents.")
        else:
            print("No new documents to process.")

    # Start chat if requested
    if args.chat:
        agent = create_agent(
            notebook_name=args.notebook_name,
            llm_provider=args.provider,
            model_name=args.model,
            temperature=args.temperature,
            top_k=args.top_k,
            max_history_length=args.max_history
        )

        # Load history if requested
        if args.load_history:
            agent.load_chat_history(args.load_history)
            print(f"Loaded chat history from {args.load_history}")

        chat_loop(agent, args.save_history)

if __name__ == "__main__":
    main()
