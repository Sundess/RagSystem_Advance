📁 Folder Structure:

RagSystem/
├── main.py # Main execution file
├── utils/ # Utility modules
│ ├── **init**.py # Package initialization
│ ├── pinecone_manager.py # Pinecone operations
│ ├── gemini_chat.py # Gemini AI integration
│ └── file_processor.py # File handling (TXT, PDF, DOCX, MD)
├── data/ # Your documents folder
├── .env # Your API keys
├── .env.example # Template for environment variables
└── requirements.txt # Dependencies

🚀 Key Features:
File Processing: Supports TXT, PDF, DOCX, and Markdown files
Pinecone Integration: Smart vector storage with status monitoring
Gemini AI: Context-aware question answering
Interactive Chat: Continuous Q&A loop
Modular Design: Clean, organized code structure

🔧 Supported File Types:
.txt - Plain text files
.pdf - PDF documents
.docx/.doc - Word documents
.md - Markdown files

The system automatically detects file type and processes accordingly. It creates an interactive chat where you can ask questions about your documents, and Gemini provides contextual answers based on the retrieved information!
