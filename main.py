import streamlit as st
import os
from dotenv import load_dotenv
from utils.pinecone_manager import PineconeManager
from utils.gemini_chat import GeminiChat
from utils.file_processor import FileProcessor
from utils.tool_agents import BookingAgent

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
        'embeddings_checked': False,
        'pinecone_manager': None,  # Cache the manager
        'gemini_chat': None,      # Cache gemini chat
        'file_processor': None    # Cache file processor
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_cached_components():
    """Get cached components to prevent reinitialization"""
    pinecone_manager = PineconeManager()
    gemini_chat = GeminiChat()
    file_processor = FileProcessor()
    return pinecone_manager, gemini_chat, file_processor


def get_components():
    """Get components from cache or session state"""
    # Use cached versions first
    if (not st.session_state.pinecone_manager or
        not st.session_state.gemini_chat or
            not st.session_state.file_processor):

        print("🔄 Getting fresh components...")
        pinecone_manager, gemini_chat, file_processor = get_cached_components()

        # Store in session state to prevent re-creation
        st.session_state.pinecone_manager = pinecone_manager
        st.session_state.gemini_chat = gemini_chat
        st.session_state.file_processor = file_processor

        # Create booking agent with the gemini chat
        st.session_state.booking_agent = BookingAgent(gemini_chat)

    return (st.session_state.pinecone_manager,
            st.session_state.gemini_chat,
            st.session_state.file_processor,
            st.session_state.booking_agent)


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
    """Enhanced intent detection"""
    # Check if booking is already active
    if st.session_state.booking_agent and st.session_state.booking_agent.is_booking_active():
        print(
            f"🔄 Booking already active, treating as booking input: '{query}'")
        return "booking"

    # Check for new booking keywords
    booking_keywords = [
        'book', 'schedule', 'appointment', 'meeting', 'callback',
        'call me', 'arrange', 'set up', 'reserve', 'contact me'
    ]

    query_lower = query.lower()
    is_booking = any(keyword in query_lower for keyword in booking_keywords)

    intent = "booking" if is_booking else "document_query"
    print(f"🔍 Intent detected: {intent} for query: '{query}' (booking_active: {st.session_state.booking_agent.is_booking_active() if st.session_state.booking_agent else False})")

    return intent


def handle_query(query, gemini_chat, pinecone_manager, booking_agent):
    """Handle user query with improved routing"""
    intent = detect_intent(query)

    if intent == "booking":
        print(f"📞 Processing booking request: '{query}'")

        try:
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

        except Exception as e:
            print(f"❌ Booking error: {e}")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"❌ Error processing booking: {str(e)}"
            })

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
            try:
                stats = pinecone_manager.get_stats()
                print(f"📊 Stats: {stats}")
                st.metric("Embeddings", stats['total_vectors'])
                st.success("Ready for queries")
            except Exception as e:
                print(f"❌ Error getting stats: {e}")
                st.warning("Stats unavailable")
        else:
            st.warning("No embeddings")

        # Actions
        st.header("🛠️ Actions")
        if st.button("Clear All"):
            try:
                pinecone_manager.clear_index()
                file_processor.clear_all_files()
                st.session_state.vectorstore = None
                st.session_state.chat_history = []
                st.success("Cleared!")
            except Exception as e:
                st.error(f"Error clearing: {e}")

        # Booking status
        if (st.session_state.booking_agent and
                st.session_state.booking_agent.is_booking_active()):
            st.warning("📋 Booking in progress")
            if st.button("Cancel Booking"):
                try:
                    msg = st.session_state.booking_agent.cancel_booking()
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": msg})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error cancelling booking: {e}")

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

    # Get components - this now uses caching
    pinecone_manager, gemini_chat, file_processor, booking_agent = get_components()

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

    # Show booking status prominently with progress
    if booking_agent and booking_agent.is_booking_active():
        progress_info = booking_agent.get_booking_progress()
        st.info(f"📋 **Booking in Progress** - {progress_info}")
        st.markdown("*Please provide the requested information to continue*")

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
        placeholder_text = ("Continue with booking..." if (booking_agent and booking_agent.is_booking_active())
                            else "Ask a question or book an appointment:")
        query = st.text_input(placeholder_text)
        submit = st.form_submit_button("Send", type="primary")

    # Handle query
    if submit and query.strip():
        st.session_state.chat_history.append(
            {"role": "user", "content": query.strip()})
        handle_query(query.strip(), gemini_chat,
                     pinecone_manager, booking_agent)
        st.rerun()  # Force UI update

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            # Also cancel any active booking
            if booking_agent and booking_agent.is_booking_active():
                booking_agent.cancel_booking()
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
    if booking_agent and booking_agent.is_booking_active():
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(
                    f"📋 **Booking Status**: {booking_agent.current_booking} booking in progress")
            with col2:
                if st.button("❌ Cancel", key="cancel_main"):
                    cancel_msg = booking_agent.cancel_booking()
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": cancel_msg})
                    st.rerun()


if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("❌ Missing API keys in .env file")
        st.stop()

    main()
