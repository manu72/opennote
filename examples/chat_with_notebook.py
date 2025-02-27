#!/usr/bin/env python3
"""
Example script demonstrating how to use the OpenNote agent programmatically.

This script shows how to:
1. Initialize an agent for a specific notebook
2. Ask questions and get responses
3. Use the agent in both interactive and non-interactive modes

Usage:
    python examples/chat_with_notebook.py notebook_name [--interactive] [--provider {openai,ollama}]
"""
import sys
import argparse
from pathlib import Path

# Add the project root to the path so we can import the opennote package
sys.path.append(str(Path(__file__).parent.parent))

from src.opennote.agent import create_agent

def main():
    """Main function to demonstrate using the OpenNote agent programmatically."""
    parser = argparse.ArgumentParser(description="Chat with an OpenNote notebook")
    parser.add_argument("notebook_name", help="Name of the notebook to chat with")
    parser.add_argument("--interactive", "-i", action="store_true", 
                        help="Start an interactive chat session")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                        help="LLM provider to use (default: openai)")
    parser.add_argument("--model", help="Model name to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for response generation (default: 0.7)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")

    args = parser.parse_args()

    # Check if the notebook exists
    notebook_path = Path(f"notebooks/{args.notebook_name}")
    if not notebook_path.exists() or not (notebook_path / "metadata.json").exists():
        print(f"Error: Notebook '{args.notebook_name}' does not exist")
        sys.exit(1)

    # Check if the notebook has been processed
    chroma_path = notebook_path / "chromadb"
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        print(f"Error: Notebook '{args.notebook_name}' has not been processed yet")
        print("Please process PDFs first using the main script or Streamlit app")
        sys.exit(1)

    print(f"Initializing agent for notebook '{args.notebook_name}'...")
    if not args.model:
        print("Error: Model name must be provided")
        sys.exit(1)
    agent = create_agent(
        notebook_name=args.notebook_name,
        llm_provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        top_k=args.top_k
    )
    print("Agent initialized successfully")

    if args.interactive:
        # Interactive mode
        print("\n=== Interactive Chat Mode ===")
        print("Type 'exit', 'quit', or 'q' to end the session")
        print("Type 'clear' to clear the conversation history")
        print("Type 'save' to save the conversation history")
        print("Type 'help' to see these commands again")
        print("===========================\n")

        while True:
            try:
                user_input = input("\nYou: ")

                # Check for special commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Ending chat session")
                    break
                elif user_input.lower() == "clear":
                    agent.clear_history()
                    print("Conversation history cleared")
                    continue
                elif user_input.lower() == "save":
                    history_path = agent.save_chat_history()
                    print(f"Conversation history saved to {history_path}")
                    continue
                elif user_input.lower() == "help":
                    print("\n=== Commands ===")
                    print("exit, quit, q: End the session")
                    print("clear: Clear the conversation history")
                    print("save: Save the conversation history")
                    print("help: Show this help message")
                    print("===============\n")
                    continue
                elif not user_input.strip():
                    continue

                # Get response from agent
                print("\nAgent: ", end="", flush=True)
                response = agent.chat(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\nEnding chat session")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
    else:
        # Non-interactive mode with example questions
        example_questions = [
            "What are the main topics covered in these documents?",
            "Can you summarize the key points from the documents?",
            "What are the most important concepts mentioned in the documents?"
        ]

        print("\n=== Example Questions ===")
        for i, question in enumerate(example_questions, 1):
            print(f"\nQuestion {i}: {question}")
            print(f"Answer: {agent.chat(question)}")

        print("\nTo chat interactively, run with the --interactive flag")

if __name__ == "__main__":
    main()
