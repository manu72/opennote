"""
Command-line interface for interacting with the OpenNote agent.
"""
import os
import argparse
import sys
from typing import Optional
from datetime import datetime

from opennote.agent import create_agent
from opennote.notebook_manager import create_notebook

def chat_loop(agent, save_history: bool = False, history_path: Optional[str] = None):
    """
    Interactive chat loop with the agent.
    
    Args:
        agent: The initialized agent
        save_history: Whether to save chat history
        history_path: Path to save chat history
    """
    print(f"Chat with {agent.notebook_name} notebook using {agent.llm_provider} ({agent.model_name})")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("Type 'save' to save the conversation history.")
    print("Type 'clear' to clear the conversation memory.")
    print("Type 'help' to see these commands again.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                break
                
            if user_input.lower() == "save":
                saved_path = agent.save_chat_history(history_path)
                print(f"Chat history saved to {saved_path}")
                continue
                
            if user_input.lower() == "clear":
                agent.conversation_memory = []
                print("Conversation memory cleared. The AI will not remember previous exchanges in this session.")
                continue
                
            if user_input.lower() == "help":
                print("Commands:")
                print("  exit, quit, q - End the conversation")
                print("  save - Save the conversation history")
                print("  clear - Clear the conversation memory")
                print("  help - Show this help message")
                continue
                
            if not user_input:
                continue
                
            response = agent.chat(user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Save history at the end if requested
    if save_history:
        saved_path = agent.save_chat_history(history_path)
        print(f"Chat history saved to {saved_path}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="OpenNote AI Agent CLI")
    parser.add_argument("notebook", help="Name of the notebook to use")
    parser.add_argument("--provider", "-p", choices=["openai", "ollama"], default="openai",
                        help="LLM provider to use (default: openai)")
    parser.add_argument("--model", "-m", help="Model name to use (defaults to environment settings)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                        help="Temperature for response generation (default: 0.7)")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--max-history", type=int, default=10,
                        help="Maximum conversation turns to remember (default: 10)")
    parser.add_argument("--create", "-c", action="store_true",
                        help="Create the notebook if it doesn't exist")
    parser.add_argument("--save-history", "-s", action="store_true",
                        help="Save chat history at the end of the session")
    parser.add_argument("--history-path", 
                        help="Path to save chat history (default: notebooks/<notebook>/history/chat_history_<timestamp>.json)")
    parser.add_argument("--load-history", "-l", 
                        help="Path to load chat history from")
    
    args = parser.parse_args()
    
    # Create notebook if requested
    if args.create:
        create_notebook(args.notebook)
    
    # Check if notebook exists
    notebook_path = os.path.join("notebooks", args.notebook)
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook '{args.notebook}' does not exist.")
        print("Use --create or -c flag to create it.")
        sys.exit(1)
    
    # Create agent
    agent = create_agent(
        notebook_name=args.notebook,
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
    
    # Start chat loop
    chat_loop(agent, args.save_history, args.history_path)

if __name__ == "__main__":
    main() 