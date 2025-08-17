# 🤖 RagSystem - AI Assistant with Document Q&A and Smart Booking

**RagSystem** is a powerful **Retrieval-Augmented Generation (RAG)** application that combines document processing with AI-powered chat capabilities. Upload documents, get intelligent answers, and book appointments or callbacks—all in one seamless interface.

## ✨ Key Features

- 📄 **Multi-Format Document Processing** - Handles TXT, PDF, DOCX, and Markdown files
- 🧠 **Smart Document Q&A** - Ask questions and get contextual answers from your documents using Google Gemini AI
- 📞 **Intelligent Booking System** - Book appointments or callbacks with natural language processing
- 🗂️ **Persistent Vector Storage** - Uses Pinecone for efficient document embeddings and retrieval
- 💬 **Interactive Chat Interface** - Streamlit-powered web interface for seamless interaction

## 🏗️ Architecture

```
RagSystem/
├── main.py                         # Main Streamlit application
├── utils/                          # Core utility modules
│   ├── __init__.py                 # Package initialization
│   ├── pinecone_manager.py         # Vector database operations
│   ├── gemini_chat.py              # Google Gemini AI integration
│   ├── file_processor.py           # Document processing (TXT/PDF/DOCX/MD)
│   ├── tool_agents.py              # Booking agents and tools
│   └── conversation_forms.py       # Form validation and processing
├── data/                           # Document storage
│   ├── raw/                        # Original uploaded files
│   └── processed/                  # Cleaned and processed documents
├── .env                            # Environment variables (API keys)
├── .env.example                    # Environment template
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **Pinecone API Key** ([Sign up here](https://www.pinecone.io/))
- **uv package manager** ([Install uv](https://docs.astral.sh/uv/getting-started/installation/))

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Sundess/RagSystem_Advance.git

cd RagSystem_Advance
```

### 2. Set up environment variables

1. Rename `.env.example` to `.env`.
2. Add your API keys in the `.env` file:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Run the application

Use `uv` to run the Streamlit app:

```bash
uv run streamlit run main.py
```

## 📖 Usage Guide

### 1. Document Processing

- **Upload Documents**: Use the sidebar to upload TXT, PDF, DOCX, or MD files
- **Auto-Processing**: Documents are automatically cleaned and vectorized
- **Persistent Storage**: Embeddings are saved in Pinecone for future sessions

### 2. Document Q&A

- **Ask Questions**: Type questions about your uploaded documents
- **Contextual Answers**: Get AI-powered responses based on document content
- **Source References**: Answers are grounded in your specific documents

### 3. Smart Booking System

- **Natural Language**: Say "book an appointment" or "call me back"
- **Guided Process**: Follow the interactive form for booking details
- **Validation**: Automatic validation of dates, times, emails, and phone numbers
- **Confirmation**: Get booking confirmations with reference numbers

### Example Interactions

```
User: "What are the main points in the uploaded document?"
AI: Based on your document, the main points are: [contextual answer]

User: "Book me an appointment for next Monday"
AI: Let's book your appointment! What's your full name?

User: "Schedule a callback please"
AI: I'll arrange a callback for you! What's your full name?
```

## 🔧 Supported File Types

| Format   | Extension       | Description              |
| -------- | --------------- | ------------------------ |
| Text     | `.txt`          | Plain text files         |
| PDF      | `.pdf`          | PDF documents            |
| Word     | `.docx`, `.doc` | Microsoft Word documents |
| Markdown | `.md`           | Markdown files           |

## 🛠️ Features in Detail

### Document Processing

- **AI-Powered Cleaning**: Uses Gemini AI to clean and structure document content
- **Chunk Splitting**: Intelligent text splitting for optimal retrieval
- **Embedding Generation**: Creates high-quality embeddings using Google's text-embedding-004 model

### Vector Database

- **Pinecone Integration**: Scalable vector storage and similarity search
- **Persistent Embeddings**: Documents remain available across sessions
- **Efficient Retrieval**: Fast similarity search for relevant context

### Booking System

- **Form Validation**: Pydantic-based validation for all input fields
- **Natural Language Processing**: LLM-powered parsing for dates and times
- **Progress Tracking**: Visual progress indicators during booking
- **Reference Numbers**: Unique booking IDs for tracking

### User Interface

- **Streamlit Framework**: Modern, responsive web interface
- **Real-time Updates**: Live chat interface with immediate responses
- **Progress Indicators**: Visual feedback during processing
- **Error Handling**: Graceful error messages and recovery

### Performance Optimization

- **Large Documents**: For documents >50MB, consider splitting them into smaller files
- **Memory Usage**: Restart the application if you notice high memory usage
- **API Limits**: Be aware of API rate limits for Gemini and Pinecone

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
