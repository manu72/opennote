#!/usr/bin/env python3
"""
Example script demonstrating how to use the OpenNote agent programmatically.
"""
import os
import sys
import argparse
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.opennote.agent import create_agent
from src.opennote.notebook_manager import create_notebook
from src.opennote.vector_store import store_text_in_chromadb, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def main():
    """Main function to demonstrate agent usage."""
    parser = argparse.ArgumentParser(description="Chat with an OpenNote notebook")
    parser.add_argument("notebook_name", help="Name of the notebook to chat with")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai",
                        help="LLM provider to use (default: openai)")
    parser.add_argument("--model", help="Model name to use (defaults to environment settings)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for response generation (default: 0.7)")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--max-history", type=int, default=10,
                        help="Maximum conversation turns to remember (default: 10)")
    parser.add_argument("--create", "-c", action="store_true",
                        help="Create the notebook if it doesn't exist")
    parser.add_argument("--load-history", "-l", 
                        help="Path to load chat history from")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start an interactive chat session after examples")
    
    args = parser.parse_args()
    
    # Create notebook if requested
    if args.create:
        create_notebook(args.notebook_name)
        print(f"Created notebook: {args.notebook_name}")
        print("Add PDF files to notebooks/{}/docs/ and run the processing script before chatting.".format(args.notebook_name))
        return
    
    # Check if notebook exists
    notebook_path = os.path.join("notebooks", args.notebook_name)
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook '{args.notebook_name}' does not exist.")
        print("Use --create or -c flag to create it.")
        return
    
    # Create agent
    agent = create_agent(
        notebook_name=args.notebook_name,
        llm_provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        max_history_length=args.max_history
    )
    
    # Load history if requested
    if args.load_history and os.path.exists(args.load_history):
        agent.load_chat_history(args.load_history)
        print(f"Loaded chat history from {args.load_history}")
    
    # Example queries demonstrating conversation memory
    example_queries = [
        "What is this notebook about?",
        "Summarize the key points from the documents.",
        "Can you elaborate on the previous summary?",  # This will use conversation memory
        "What are the main topics covered in these documents?",
        "How do those topics relate to what we discussed earlier?"  # This will use conversation memory
    ]
    
    print(f"Chatting with notebook: {args.notebook_name}")
    print(f"Using {args.provider} as the LLM provider")
    print(f"Retrieving top {args.top_k} chunks for each query")
    print(f"Remembering up to {args.max_history} conversation turns")
    print("\nExample queries demonstrating conversation memory:")
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nQuery {i}: {query}")
        response = agent.chat(query)
        print(f"Response: {response}")
        print("-" * 50)
    
    # Save chat history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(notebook_path, "history", f"example_chat_{timestamp}.json")
    os.makedirs(os.path.join(notebook_path, "history"), exist_ok=True)
    agent.save_chat_history(history_path)
    print(f"\nChat history saved to: {history_path}")
    
    # Interactive mode if requested
    if args.interactive:
        print("\nStarting interactive chat mode. Type 'exit', 'quit', or 'q' to end.")
        print("Type 'save' to save the conversation history.")
        print("Type 'clear' to clear the conversation memory.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                    
                if user_input.lower() == "save":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(notebook_path, "history", f"interactive_chat_{timestamp}.json")
                    agent.save_chat_history(save_path)
                    print(f"Chat history saved to {save_path}")
                    continue
                    
                if user_input.lower() == "clear":
                    agent.conversation_memory = []
                    print("Conversation memory cleared.")
                    continue
                    
                if not user_input:
                    continue
                    
                response = agent.chat(user_input)
                print(f"\nAI: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Save final history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = os.path.join(notebook_path, "history", f"final_chat_{timestamp}.json")
        agent.save_chat_history(final_path)
        print(f"\nFinal chat history saved to: {final_path}")

if __name__ == "__main__":
    main() 