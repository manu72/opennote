"""
This module implements a RAG-based AI agent that can chat with users
using the vector database as a knowledge base.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union
import requests
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import our vector store functions
from opennote.vector_store import query_vector_db

# Load environment variables
load_dotenv()

# Constants
CHROMA_EMBEDDING_MODEL = os.getenv("CHROMA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-coder:latest")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5  # Number of relevant chunks to retrieve
MAX_HISTORY_LENGTH = 10  # Maximum number of conversation turns to keep in memory

class Agent:
    """
    RAG-based AI agent that uses vector database for knowledge retrieval
    and LLMs for generating responses.
    """
    
    def __init__(
        self, 
        notebook_name: str,
        llm_provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        max_history_length: int = MAX_HISTORY_LENGTH
    ):
        """
        Initialize the agent with the specified notebook and LLM configuration.
        
        Args:
            notebook_name: Name of the notebook to use as knowledge base
            llm_provider: 'openai' or 'ollama'
            model_name: Model to use (defaults to environment variables if not specified)
            temperature: Temperature for response generation
            top_k: Number of relevant chunks to retrieve from vector DB
            max_history_length: Maximum number of conversation turns to keep in memory
        """
        self.notebook_name = notebook_name
        self.llm_provider = llm_provider.lower()
        self.temperature = temperature
        self.top_k = top_k
        self.max_history_length = max_history_length
        
        # Set model based on provider
        if model_name:
            self.model_name = model_name
        elif self.llm_provider == "openai":
            self.model_name = OPENAI_MODEL
        else:
            self.model_name = OLLAMA_MODEL
            
        # Chat history
        self.chat_history = []
        
        # Conversation memory
        self.conversation_memory = []
    
    def _retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from the vector database.
        
        Args:
            query: User query to find relevant information for
            
        Returns:
            List of relevant text chunks with metadata
        """
        # Use the improved vector_store query function
        return query_vector_db(self.notebook_name, query, self.top_k)
    
    def _generate_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a prompt for the LLM using the retrieved context and conversation history.
        
        Args:
            query: User query
            context: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Format context information
        context_str = "\n\n".join([
            f"Document: {doc['metadata'].get('source', 'Unknown')} (Relevance: {doc['relevance_score']:.2f})\n{doc['text']}" 
            for doc in context
        ])
        
        # Format conversation history
        history_str = ""
        if self.conversation_memory:
            history_str = "Previous conversation:\n"
            for i, exchange in enumerate(self.conversation_memory):
                history_str += f"User: {exchange['query']}\nAI: {exchange['response']}\n\n"
        
        # Build the complete prompt
        prompt = f"""You are an AI assistant for answering questions based on the provided documents.

CONTEXT INFORMATION:
{context_str}

{history_str}
Based on the context information provided above and our conversation history, answer the following question. 
If the answer cannot be determined from the context, say "I don't have enough information to answer that question."

USER QUESTION: {query}

ANSWER:"""
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API to generate a response.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated response
        """
        try:
            import openai
            
            if not OPENAI_API_KEY:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")
            
            openai.api_key = OPENAI_API_KEY
            
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API to generate a response.
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated response
        """
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error calling Ollama API: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"
    
    def chat(self, query: str) -> str:
        """
        Process a user query and generate a response using RAG.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        # Retrieve relevant context
        context = self._retrieve_context(query)
        
        # Generate prompt with context and conversation memory
        prompt = self._generate_prompt(query, context)
        
        # Generate response based on provider
        if self.llm_provider == "openai":
            response = self._call_openai(prompt)
        else:  # ollama
            response = self._call_ollama(prompt)
        
        # Update chat history
        timestamp = datetime.now().isoformat()
        self.chat_history.append({
            "timestamp": timestamp,
            "query": query, 
            "response": response,
            "context": [{"text": doc["text"], "metadata": doc["metadata"]} for doc in context]
        })
        
        # Update conversation memory (limited to max_history_length)
        self.conversation_memory.append({"query": query, "response": response})
        if len(self.conversation_memory) > self.max_history_length:
            self.conversation_memory = self.conversation_memory[-self.max_history_length:]
        
        return response
    
    def save_chat_history(self, filepath: Optional[str] = None) -> None:
        """
        Save the chat history to a file.
        
        Args:
            filepath: Path to save the chat history. If None, saves to the notebook directory.
        """
        if not filepath:
            notebook_path = os.path.join("notebooks", self.notebook_name)
            history_dir = os.path.join(notebook_path, "history")
            
            # Create history directory if it doesn't exist
            os.makedirs(history_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(history_dir, f"chat_history_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.chat_history, f, indent=4)
            
        return filepath
    
    def load_chat_history(self, filepath: str) -> None:
        """
        Load chat history from a file.
        
        Args:
            filepath: Path to load the chat history from
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.chat_history = json.load(f)
                
            # Rebuild conversation memory from chat history
            self.conversation_memory = [
                {"query": item["query"], "response": item["response"]} 
                for item in self.chat_history[-self.max_history_length:]
            ]


def create_agent(
    notebook_name: str,
    llm_provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    max_history_length: int = MAX_HISTORY_LENGTH
) -> Agent:
    """
    Factory function to create an agent instance.
    
    Args:
        notebook_name: Name of the notebook to use
        llm_provider: 'openai' or 'ollama'
        model_name: Model name to use
        temperature: Temperature for response generation
        top_k: Number of relevant chunks to retrieve
        max_history_length: Maximum number of conversation turns to keep in memory
        
    Returns:
        Configured Agent instance
    """
    return Agent(
        notebook_name=notebook_name,
        llm_provider=llm_provider,
        model_name=model_name,
        temperature=temperature,
        top_k=top_k,
        max_history_length=max_history_length
    ) 