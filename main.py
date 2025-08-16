import streamlit as st
import os
from dotenv import load_dotenv
from utils.pinecone_manager import PineconeManager
from utils.gemini_chat import GeminiChat
from utils.file_processor import FileProcessor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System with Pinecone & Gemini",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def initialize_components():
    """Initialize and cache components"""
    pinecone_manager = PineconeManager()
    gemini_chat = GeminiChat()
    file_processor = FileProcessor()
    return pinecone_manager, gemini_chat, file_processor


def main():
    """
    Main Streamlit application
    """
    st.title("🤖 RAG System with Pinecone & Gemini")
    st.markdown("Upload documents and ask questions using AI-powered retrieval!")

    # Initialize components
    pinecone_manager, gemini_chat, file_processor = initialize_components()

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=['txt', 'pdf', 'docx', 'md'],
            help="Supported formats: TXT, PDF, DOCX, MD"
        )

        process_button = st.button("🚀 Process Document", type="primary")

        # Display processing status
        status_container = st.container()

        # Clear buttons section
        st.header("🗑️ Clear Data")

        # Clear embeddings and files
        if st.button("🗂️ Clear All Data", type="secondary", help="Clear vector database and all files"):
            if st.session_state.get('confirm_clear', False):
                with st.spinner("Clearing all data..."):
                    try:
                        # Clear vector database
                        if 'vectorstore' in st.session_state and st.session_state.vectorstore:
                            pinecone_manager.clear_index()

                        # Clear files
                        file_processor.clear_all_files()

                        # Reset session state
                        st.session_state.vectorstore = None
                        st.session_state.chat_history = []

                        st.success("✅ All data cleared successfully!")
                        st.session_state.confirm_clear = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error clearing data: {str(e)}")
            else:
                st.session_state.confirm_clear = True
                st.warning(
                    "⚠️ This will delete all embeddings and files. Click again to confirm.")

        # Reset confirmation if other buttons are clicked
        if 'confirm_clear' in st.session_state and st.session_state.confirm_clear:
            if st.button("❌ Cancel Clear"):
                st.session_state.confirm_clear = False
                st.rerun()

        status_container = st.container()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")

        # Chat container
        chat_container = st.container()

        # Query input
        query = st.text_input(
            "Ask a question about your documents:", key="query_input")
        ask_button = st.button("Ask Question", type="secondary")

    with col2:
        st.header("📊 System Status")
        if st.session_state.vectorstore:
            st.success("✅ Vector database ready")
        else:
            st.warning("⏳ No documents processed yet")

        if uploaded_file:
            st.info(f"📄 File ready: {uploaded_file.name}")

        # Show file statistics
        try:
            file_stats = file_processor.get_directory_stats()
            st.metric("Raw files", file_stats['raw_count'])
            st.metric("Processed files", file_stats['processed_count'])
        except:
            pass

    # Process document when button is clicked
    if process_button and uploaded_file:
        with status_container:
            with st.spinner("Processing document..."):
                try:
                    # Step 1: Save uploaded file to raw folder
                    st.info("📥 Saving uploaded file...")
                    raw_file_path = file_processor.save_uploaded_file(
                        uploaded_file)

                    # Step 2: Extract and clean text
                    st.info("🔤 Extracting and cleaning text...")
                    cleaned_content = file_processor.process_file_with_cleaning(
                        raw_file_path, gemini_chat
                    )

                    # Step 3: Setup Pinecone
                    st.info("🔧 Setting up vector database...")
                    embeddings, index_name, pc = pinecone_manager.setup_pinecone_with_google_embeddings()

                    # Step 4: Store in vector database
                    st.info("💾 Storing in vector database...")
                    vectorstore = pinecone_manager.process_and_store_documents(
                        cleaned_content, embeddings, index_name, pc
                    )

                    # Update session state
                    st.session_state.vectorstore = vectorstore

                    st.success("✅ Document processed successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

    # Handle questions
    if (ask_button or query) and query and st.session_state.vectorstore:
        with chat_container:
            # Add user question to chat history
            st.session_state.chat_history.append(
                {"role": "user", "content": query})

            with st.spinner("Searching and generating answer..."):
                try:
                    # Search for relevant documents
                    relevant_docs = pinecone_manager.query_vectorstore(
                        st.session_state.vectorstore, query, k=3
                    )

                    if relevant_docs:
                        # Generate answer using Gemini with context
                        context = "\n\n".join(
                            [doc.page_content for doc in relevant_docs])
                        answer = gemini_chat.generate_answer_with_context(
                            query, context)

                        # Add AI response to chat history
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer})
                    else:
                        answer = "❌ No relevant documents found. Try rephrasing your question."
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        with chat_container:
            st.subheader("Chat History")
            for i, message in enumerate(reversed(st.session_state.chat_history)):
                if message["role"] == "user":
                    st.write(f"**🧑 You:** {message['content']}")
                else:
                    st.write(f"**🤖 AI:** {message['content']}")

                if i < len(st.session_state.chat_history) - 1:
                    st.divider()

    # Clear chat history button (fixed)
    if st.session_state.chat_history:
        col_clear1, col_clear2 = st.columns([1, 4])
        with col_clear1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, Pinecone, and Google Gemini AI")


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("❌ Missing required environment variables. Please set GOOGLE_API_KEY and PINECONE_API_KEY in your .env file.")
        st.stop()

    main()
