import streamlit as st
import os
from dotenv import load_dotenv
from utils.pinecone_manager import PineconeManager
from utils.gemini_chat import GeminiChat
from utils.file_processor import FileProcessor
from utils.tool_agents import SimplifiedBookingAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System with AI Assistant",
    page_icon="🤖",
    layout="wide"
)


def init_session_state():
    """Initialize session state"""
    defaults = {
        'vectorstore': None,
        'chat_history': [],
        'booking_agent': None,
        'embeddings_checked': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_components():
    """Get cached components"""
    pinecone_manager = PineconeManager()
    gemini_chat = GeminiChat()
    file_processor = FileProcessor()
    booking_agent = SimplifiedBookingAgent(gemini_chat)
    return pinecone_manager, gemini_chat, file_processor, booking_agent


def check_existing_embeddings(pinecone_manager):
    """Check for existing embeddings on startup"""
    if not st.session_state.embeddings_checked:
        st.session_state.embeddings_checked = True

        vectorstore, has_embeddings = pinecone_manager.setup_vectorstore()

        if has_embeddings:
            st.session_state.vectorstore = vectorstore
            st.success("✅ Found existing embeddings - Ready for queries!")
        else:
            st.info(
                "📄 No existing embeddings found. Upload documents to get started.")


def detect_intent(query):
    """Simple keyword-based intent detection"""
    booking_keywords = [
        'book', 'schedule', 'appointment', 'meeting', 'callback',
        'call me', 'arrange', 'set up', 'reserve', 'contact me'
    ]

    query_lower = query.lower()
    return "booking" if any(keyword in query_lower for keyword in booking_keywords) else "document_query"


def handle_query(query, gemini_chat, pinecone_manager, booking_agent):
    """Handle user query with simplified routing"""
    intent = detect_intent(query)

    print(f"🔍 Intent detected: {intent} for query: '{query}'")

    if intent == "booking":
        # Handle booking
        print(f"📞 Processing booking with agent: {booking_agent}")
        response, complete = booking_agent.process_message(query)

        print(f"📞 Booking response: {response}")
        print(f"📞 Booking complete: {complete}")

        if response:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})
            if complete:
                st.balloons()
        else:
            # Fallback response
            fallback = "I can help you book an appointment! Please tell me what you need."
            st.session_state.chat_history.append(
                {"role": "assistant", "content": fallback})

    else:
        # Handle document queries
        if st.session_state.vectorstore:
            with st.spinner("Searching documents..."):
                try:
                    docs = st.session_state.vectorstore.similarity_search(
                        query, k=3)

                    if docs:
                        context = "\n\n".join(
                            [doc.page_content for doc in docs])
                        answer = gemini_chat.generate_answer_with_context(
                            query, context)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer})
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "❌ No relevant documents found."
                        })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error searching documents: {str(e)}"
                    })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "📄 No documents available. Upload some documents first, or I can help you book an appointment!"
            })


def render_sidebar(file_processor, pinecone_manager):
    """Render sidebar"""
    with st.sidebar:
        st.header("📁 Documents")

        uploaded_file = st.file_uploader(
            "Upload document",
            type=['txt', 'pdf', 'docx', 'md']
        )

        process_btn = st.button("🚀 Process", type="primary")

        # Stats
        st.header("📊 Status")
        if st.session_state.vectorstore:
            stats = pinecone_manager.get_stats()
            print(stats)
            st.metric("Embeddings", stats['total_vectors'])
            st.success("Ready for queries")
        else:
            st.warning("No embeddings")

        # Actions
        st.header("🛠️ Actions")
        if st.button("Clear All"):
            pinecone_manager.clear_index()
            file_processor.clear_all_files()
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.success("Cleared!")

        # Booking status
        if (st.session_state.booking_agent and
                st.session_state.booking_agent.is_booking_active()):
            st.warning("📋 Booking active")
            if st.button("Cancel Booking"):
                msg = st.session_state.booking_agent.cancel_booking()
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": msg})
                st.rerun()

    return uploaded_file, process_btn


def process_document(uploaded_file, file_processor, gemini_chat, pinecone_manager):
    """Process uploaded document"""
    with st.spinner("Processing..."):
        try:
            # Process file
            raw_path = file_processor.save_uploaded_file(uploaded_file)
            content = file_processor.process_file_with_cleaning(
                raw_path, gemini_chat)

            # Add to vectorstore
            vectorstore = pinecone_manager.add_documents(content)
            st.session_state.vectorstore = vectorstore

            st.success("✅ Document processed!")
            st.balloons()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def main():
    """Main application"""
    init_session_state()

    st.title("🤖 AI Assistant")
    st.markdown("Upload documents and ask questions, or book appointments!")

    # Get components
    pinecone_manager, gemini_chat, file_processor, booking_agent = get_components()

    # Store booking agent in session state
    if not st.session_state.booking_agent:
        st.session_state.booking_agent = booking_agent

    # Check existing embeddings
    check_existing_embeddings(pinecone_manager)

    # Sidebar
    uploaded_file, process_btn = render_sidebar(
        file_processor, pinecone_manager)

    # Process document
    if process_btn and uploaded_file:
        process_document(uploaded_file, file_processor,
                         gemini_chat, pinecone_manager)

    # Chat interface
    st.header("💬 Chat")

    # Show latest response above input
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        if latest["role"] == "assistant":
            st.markdown("### 🤖 Response:")
            st.markdown(latest["content"])
            st.caption(
                f"💭 {len(st.session_state.chat_history)} messages in conversation")
            st.divider()

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input("Ask a question or book an appointment:")
        submit = st.form_submit_button("Send", type="primary")

    # Handle query
    if submit and query.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": query.strip()})
        handle_query(query.strip(), gemini_chat, pinecone_manager,
                     st.session_state.booking_agent)
        st.rerun()  # This is crucial - forces UI update

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat cleared!")

    with col2:
        if st.button("📋 Show History"):
            if st.session_state.chat_history:
                with st.expander("💬 Full History", expanded=True):
                    for msg in st.session_state.chat_history:
                        role_icon = "🧑" if msg["role"] == "user" else "🤖"
                        st.markdown(
                            f"**{role_icon} {msg['role'].title()}:** {msg['content']}")

    # Booking status in main area
    if st.session_state.booking_agent and st.session_state.booking_agent.is_booking_active():
        st.info("📋 Booking in progress... Please provide the requested information.")
        if st.button("❌ Cancel Current Booking"):
            cancel_msg = st.session_state.booking_agent.cancel_booking()
            st.session_state.chat_history.append(
                {"role": "assistant", "content": cancel_msg})
            st.rerun()


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("❌ Missing API keys in .env file")
        st.stop()

    main()
