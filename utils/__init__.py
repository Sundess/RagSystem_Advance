"""
RAG System Utils Package - Simplified Version

This package contains utility modules for the RAG (Retrieval Augmented Generation) system:

- pinecone_manager: Handles Pinecone vector database operations with persistent embeddings
- gemini_chat: Manages Google Gemini AI integration  
- file_processor: Processes various file formats (TXT, PDF, DOCX, MD)
- langchain_booking_tools: LangChain-based booking tools and simplified agent
"""

from .pinecone_manager import PineconeManager
from .gemini_chat import GeminiChat
from .file_processor import FileProcessor
from .tool_agents import SimplifiedBookingAgent, BookingTool

__all__ = [
    'PineconeManager',
    'GeminiChat',
    'FileProcessor',
    'SimplifiedBookingAgent',
    'BookingTool'
]

# Version info
__version__ = "2.0.0"
__description__ = "Simplified RAG System with LangChain tools and persistent embeddings"
