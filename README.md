# 📖 RagSystem

RagSystem is a **Retrieval-Augmented Generation (RAG)** application that lets you upload documents, process them into embeddings and store them in pineocne vector database, and interact with them through **Gemini AI** for context-aware Q\&A.

---

## 📂 Folder Structure

```
RagSystem/
├── main.py                    # Main entry point
├── utils/                     # Utility modules
│   ├── __init__.py            # Package initialization
│   ├── pinecone_manager.py    # Pinecone vector DB operations
│   ├── gemini_chat.py         # Gemini AI integration
│   └── file_processor.py      # File handling (TXT, PDF, DOCX, MD)
├── data/                      # Documents folder
├── .env                       # Environment variables (API keys)
├── .env.example               # Template for environment setup
└── requirements.txt           # Project dependencies
```

---

## 🚀 Features

* **File Processing**: Automatically handles `.txt`, `.pdf`, `.docx`, and `.md`
* **Pinecone Integration**: Vector storage + retrieval with status monitoring
* **Gemini AI**: Contextual, accurate answers to your queries
* **Interactive Chat**: Ask follow-up questions in a continuous conversation
* **Modular Design**: Clean, extensible, and easy-to-maintain codebase

---

## 🔧 Supported File Types

* `.txt` → Plain text
* `.pdf` → PDF documents
* `.docx` / `.doc` → Word files
* `.md` → Markdown files

The system **auto-detects file type** and processes it seamlessly. Upload documents, chat with Gemini, and get precise, context-driven answers instantly.

Do you also want me to add a **Quick Start section with setup and usage commands** (like `pip install -r requirements.txt` and `python main.py`)? That would make the README even more practical.
